import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Model as PiModel,
  type ModelThinkingLevel as PiModelThinkingLevel,
  type ProviderStreamOptions,
  type ProviderStreams,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { record_http_response_status } from "../network/http-response-status";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import { googleGenerativeAIApi } from "@earendil-works/pi-ai/api/google-generative-ai.lazy";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";

import { DEFAULT_MODEL_AGENT_CONFIG } from "../../domain/model-agent";
import { AppError } from "../../shared/error";
import {
  apply_one_shot_request_overrides,
  resolve_one_shot_generation_options,
} from "./llm-client-policy";
import type { LLMMessage } from "./llm-types";
import { resolve_model_capability, resolve_pi_thinking_level } from "./model-capability";
import type { ModelRequestSnapshot } from "./policy/policy-types";

// Pi provider 身份只用于 adapter 与 ModelRuntime 注册，项目策略直接使用 api_format。
type PiApi =
  | "openai-completions"
  | "openai-responses"
  | "anthropic-messages"
  | "google-generative-ai";
type PiProvider = "openai" | "openai-compatible" | "anthropic" | "google";
const ANTHROPIC_FALLBACK_MAX_TOKENS = 64_000; // 缺少模型规格时仍满足 Messages API 必填上限

/** 统一 OneShot 调用形状，A/G 通过它转接 Pi 的 streamSimple。 */
type OneShotStream = (
  model: PiModel<PiApi>,
  context: Context,
  options?: ProviderStreamOptions,
) => AssistantMessageEventStream;

/** 调用方可覆盖显示身份与容量，缺省容量沿用统一模型规格，协议字段由本模块补齐。 */
type PiModelSettings = Readonly<{
  name: string;
  contextWindow?: number;
  maxTokens?: number;
  fallbackMaxTokens?: number; // 只在调用方和模型规格均未提供容量时使用
  input: PiModel<PiApi>["input"];
}>;

/** OneShot 与 Agent 共用同一次策略解析生成 Pi 能力映射和实际思考档位。 */
export function resolve_pi_model(
  snapshot: ModelRequestSnapshot,
  settings: PiModelSettings,
): {
  model: PiModel<PiApi>;
  thinkingLevel: PiModelThinkingLevel;
  stream: ProviderStreams["stream"];
  streamSimple: ProviderStreams["streamSimple"];
} {
  const api = resolve_pi_api(snapshot.api_format);
  const capability = resolve_model_capability({
    api_format: snapshot.api_format,
    model_id: snapshot.model_id,
    agent: DEFAULT_MODEL_AGENT_CONFIG,
  });
  const thinking_level = resolve_pi_thinking_level(
    snapshot.thinking_level,
    capability.available_thinking_levels,
  );
  const compat = {
    ...capability.compat,
    // 自定义 OpenAI-compatible 服务只共同保证 system role；OneShot 会继续冻结旧 payload 形状。
    ...(api.api === "openai-completions" ? { supportsDeveloperRole: false } : {}),
  };
  const model: PiModel<PiApi> = {
    id: snapshot.model_id,
    name: settings.name,
    provider: api.provider,
    api: api.api,
    baseUrl: snapshot.base_url,
    reasoning: capability.reasoning,
    ...(capability.thinking_level_map === undefined
      ? {}
      : { thinkingLevelMap: { ...capability.thinking_level_map } }),
    input: settings.input,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: settings.contextWindow ?? capability.context_window ?? 0,
    maxTokens: settings.maxTokens ?? capability.max_tokens ?? settings.fallbackMaxTokens ?? 0,
    ...(Object.keys(compat).length === 0 ? {} : { compat }),
  };
  return {
    model,
    thinkingLevel: thinking_level,
    stream: api.stream,
    streamSimple: api.streamSimple,
  };
}

/** 组装一次 OneShot Pi 请求；应用级超时仍由 LLMClient 独立拥有。 */
export function resolve_one_shot_pi_request(
  snapshot: ModelRequestSnapshot,
  messages: LLMMessage[],
  signal: AbortSignal,
): {
  model: PiModel<PiApi>;
  context: Context;
  options: ProviderStreamOptions;
  stream: OneShotStream;
} {
  const generation = resolve_one_shot_generation_options(snapshot);
  const resolved = resolve_pi_model(snapshot, {
    name: snapshot.model_id,
    // Anthropic 要求 max_tokens：显式值冻结总 ceiling，自动值使用模型规格或未知模型回退。
    ...(snapshot.api_format !== "Anthropic"
      ? {}
      : generation.maxTokens === undefined
        ? { fallbackMaxTokens: ANTHROPIC_FALLBACK_MAX_TOKENS }
        : { maxTokens: generation.maxTokens }),
    input: ["text"],
  });
  // Chat Completions 保持既有 payload；Responses 直接使用 Pi 的原生 Items 与 store:false 契约。
  const model: PiModel<PiApi> =
    resolved.model.api === "openai-completions"
      ? {
          ...resolved.model,
          compat: {
            ...resolved.model.compat,
            supportsDeveloperRole: false,
            supportsStore: false,
            supportsUsageInStreaming: true,
            maxTokensField: "max_tokens",
          },
        }
      : resolved.model;
  const options: ProviderStreamOptions = {
    apiKey: snapshot.api_keys[0] ?? "no_key_required",
    cacheRetention: "none",
    headers: { ...snapshot.headers },
    maxRetries: 0,
    signal,
    ...(generation.temperature === undefined ? {} : { temperature: generation.temperature }),
    ...(generation.maxTokens === undefined ? {} : { maxTokens: generation.maxTokens }),
    ...((snapshot.api_format === "OpenAI" || snapshot.api_format === "OpenAIResponses") &&
    resolved.model.reasoning &&
    resolved.thinkingLevel !== "off"
      ? { reasoningEffort: resolved.thinkingLevel }
      : {}),
    ...((snapshot.api_format === "Google" || snapshot.api_format === "Anthropic") &&
    resolved.model.reasoning &&
    resolved.thinkingLevel !== "off"
      ? { reasoning: resolved.thinkingLevel }
      : {}),
    ...(snapshot.api_format === "Anthropic" ? { interleavedThinking: false } : {}),
    onPayload: (payload) => apply_one_shot_request_overrides(snapshot, payload, signal),
  };
  const stream: OneShotStream =
    snapshot.api_format === "SakuraLLM"
      ? (active_model, context, active_options) =>
          execute_sakura_one_shot_stream(active_model, context, active_options)
      : snapshot.api_format === "Google" || snapshot.api_format === "Anthropic"
        ? (active_model, context, active_options) =>
            resolved.streamSimple(
              active_model,
              context,
              active_options as SimpleStreamOptions | undefined,
            )
        : (active_model, context, active_options) =>
            resolved.stream(active_model, context, active_options);
  return {
    model,
    context: build_pi_context(snapshot, messages),
    options,
    stream,
  };
}

/** 产品 API 枚举只在这里绑定 Pi provider 身份与惰性 adapter。 */
function resolve_pi_api(api_format: ModelRequestSnapshot["api_format"]): {
  provider: PiProvider;
  api: PiApi;
  stream: ProviderStreams["stream"];
  streamSimple: ProviderStreams["streamSimple"];
} {
  if (api_format === "SakuraLLM") {
    return {
      provider: "openai-compatible",
      api: "openai-completions",
      stream: execute_sakura_one_shot_stream,
      streamSimple: execute_sakura_one_shot_stream,
    };
  }
  if (api_format === "Anthropic") {
    return { provider: "anthropic", api: "anthropic-messages", ...anthropicMessagesApi() };
  }
  if (api_format === "Google") {
    return { provider: "google", api: "google-generative-ai", ...googleGenerativeAIApi() };
  }
  if (api_format === "OpenAIResponses") {
    return { provider: "openai", api: "openai-responses", ...openAIResponsesApi() };
  }
  return { provider: "openai", api: "openai-completions", ...openAICompletionsApi() };
}

/** 保留现有 OneShot 提示词语义：Google 把 system 当首条 user，其余协议单独传 system。 */
function build_pi_context(snapshot: ModelRequestSnapshot, messages: LLMMessage[]): Context {
  const system_prompt = messages
    .filter((message) => message.role === "system")
    .map((message) => message.content.trim())
    .filter(Boolean)
    .join("\n\n");
  const user_messages = messages
    .filter((message) => message.role === "user")
    .map((message) => message.content.trim())
    .filter(Boolean)
    .map((content) => ({ role: "user" as const, content, timestamp: 0 }));

  if (snapshot.api_format === "Google") {
    const google_messages =
      system_prompt === ""
        ? user_messages
        : [{ role: "user" as const, content: system_prompt, timestamp: 0 }, ...user_messages];
    assert_non_empty_messages(google_messages.length, snapshot.api_format);
    return { messages: google_messages };
  }
  if (snapshot.api_format === "Anthropic") {
    assert_non_empty_messages(user_messages.length, snapshot.api_format);
  } else {
    assert_non_empty_messages(
      user_messages.length + (system_prompt === "" ? 0 : 1),
      snapshot.api_format,
    );
  }
  return {
    ...(system_prompt === "" ? {} : { systemPrompt: system_prompt }),
    messages: user_messages,
  };
}

/** SakuraLLM 使用非流式 HTTP POST 请求以兼容不支持 SSE 流式的 Sakura 接口。 */
function execute_sakura_one_shot_stream(
  model: PiModel<PiApi>,
  context: Context,
  options?: ProviderStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  (async () => {
    try {
      options?.signal?.throwIfAborted();

      const messages: Array<{ role: string; content: string }> = [];
      if (context.systemPrompt && context.systemPrompt.trim() !== "") {
        messages.push({ role: "system", content: context.systemPrompt });
      }
      for (const msg of context.messages) {
        messages.push({ role: msg.role, content: msg.content });
      }

      let payload: Record<string, unknown> = {
        model: model.id,
        messages,
        stream: false,
        ...(options?.temperature !== undefined ? { temperature: options.temperature } : {}),
        ...(options?.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
      };

      if (options?.onPayload) {
        const next = await options.onPayload(payload, model);
        if (next !== undefined && typeof next === "object" && next !== null) {
          payload = next as Record<string, unknown>;
        }
      }
      payload["stream"] = false;
      delete payload["tools"];
      delete payload["tool_choice"];
      delete payload["store"];
      delete payload["stream_options"];
      delete payload["prompt_cache_key"];
      delete payload["prompt_cache_retention"];

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...options?.headers,
      };
      if (options?.apiKey && options.apiKey !== "no_key_required") {
        headers["Authorization"] = `Bearer ${options.apiKey}`;
      }

      const baseUrl = model.baseUrl.replace(/\/+$/u, "");
      const url = `${baseUrl}/chat/completions`;

      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: options?.signal,
      });

      record_http_response_status(response.status);

      if (!response.ok) {
        const error_text = (await response.text()).trim();
        const error_message = error_text
          ? `${response.status} status code (${error_text})`
          : `${response.status} status code (no body)`;
        const errMessage: AssistantMessage = {
          role: "assistant",
          content: [],
          api: "openai-completions",
          provider: "openai-compatible",
          model: model.id,
          usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: "error",
          errorMessage: error_message,
          timestamp: Date.now(),
        };
        stream.push({ type: "error", reason: "error", error: errMessage });
        stream.end();
        return;
      }

      const data = (await response.json()) as Record<string, unknown>;
      const choices = Array.isArray(data["choices"]) ? data["choices"] : [];
      const choice = (choices[0] ?? {}) as Record<string, unknown>;
      const choice_message = (choice["message"] ?? {}) as Record<string, unknown>;
      const text = String(choice_message["content"] ?? "");
      const reasoning =
        choice_message["reasoning_content"] ?? choice_message["thinking"] ?? undefined;

      const content_blocks: AssistantMessage["content"] = [];
      if (typeof reasoning === "string" && reasoning.trim() !== "") {
        content_blocks.push({ type: "thinking", thinking: reasoning });
      }
      content_blocks.push({ type: "text", text });

      const raw_usage = (data["usage"] ?? {}) as Record<string, unknown>;
      const input_tokens = Number(raw_usage["prompt_tokens"] ?? 0);
      const output_tokens = Number(raw_usage["completion_tokens"] ?? 0);
      const total_tokens = Number(raw_usage["total_tokens"] ?? input_tokens + output_tokens);

      const finish_reason = String(choice["finish_reason"] ?? "stop");
      const stopReason = finish_reason === "length" ? "length" : "stop";

      const assistantMsg: AssistantMessage = {
        role: "assistant",
        content: content_blocks,
        api: "openai-completions",
        provider: "openai-compatible",
        model: model.id,
        usage: {
          input: Number.isFinite(input_tokens) ? input_tokens : 0,
          output: Number.isFinite(output_tokens) ? output_tokens : 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: Number.isFinite(total_tokens) ? total_tokens : 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason,
        timestamp: Date.now(),
      };

      await options?.onResponse?.({ status: response.status, headers: headers_to_record(response.headers) }, model);

      stream.push({ type: "start", partial: assistantMsg });
      if (text !== "") {
        const text_index = content_blocks.findIndex((b) => b.type === "text");
        const contentIndex = text_index >= 0 ? text_index : 0;
        stream.push({ type: "text_start", contentIndex, partial: assistantMsg });
        stream.push({ type: "text_delta", contentIndex, delta: text, partial: assistantMsg });
        stream.push({ type: "text_end", contentIndex, content: text, partial: assistantMsg });
      }
      stream.push({ type: "done", reason: stopReason, message: assistantMsg });
      stream.end();
    } catch (error) {
      const is_aborted = options?.signal?.aborted;
      const error_message = error instanceof Error ? error.message : String(error);
      const errMessage: AssistantMessage = {
        role: "assistant",
        content: [],
        api: "openai-completions",
        provider: "openai-compatible",
        model: model.id,
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: is_aborted ? "aborted" : "error",
        errorMessage: is_aborted ? "请求已取消。" : error_message,
        timestamp: Date.now(),
      };
      stream.push({
        type: "error",
        reason: is_aborted ? "aborted" : "error",
        error: errMessage,
      });
      stream.end();
    }
  })();

  return stream;
}

function headers_to_record(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}

/** 空提示词在发起远端请求前按 API 格式语义转为稳定校验错误。 */
function assert_non_empty_messages(
  count: number,
  api_format: ModelRequestSnapshot["api_format"],
): void {
  if (count > 0) return;
  throw new AppError("request.validation_failed", {
    public_details: { field: "messages" },
    diagnostic_context: { api_format, reason: "empty_messages" },
  });
}
