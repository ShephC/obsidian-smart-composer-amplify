import { v4 as uuidv4 } from 'uuid'

import { ChatModel } from '../../types/chat-model.types'
import {
  LLMOptions,
  LLMRequest,
  LLMRequestNonStreaming,
  LLMRequestStreaming,
  RequestMessage,
} from '../../types/llm/request'
import {
  LLMResponseNonStreaming,
  LLMResponseStreaming,
} from '../../types/llm/response'
import { LLMProvider } from '../../types/provider.types'
import { postJson } from '../../utils/llm/httpTransport'

import { BaseLLMProvider } from './base'

const AMPLIFY_CHAT_URL = 'https://prod-api.vanderbilt.ai/chat'

type AmplifyChatResponse = {
  success: boolean
  message?: string | string[]
  data?: unknown
}

type AmplifyMessage = {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export class AmplifyProvider extends BaseLLMProvider<
  Extract<LLMProvider, { type: 'amplify' }>
> {
  async generateResponse(
    model: ChatModel,
    request: LLMRequestNonStreaming,
    options?: LLMOptions,
  ): Promise<LLMResponseNonStreaming> {
    return this.sendRequest(model, request, options)
  }

  async streamResponse(
    model: ChatModel,
    request: LLMRequestStreaming,
    options?: LLMOptions,
  ): Promise<AsyncIterable<LLMResponseStreaming>> {
    const response = await this.sendRequest(model, request, options)
    return this.singleChunkStream(response)
  }

  async getEmbedding(
    _model: string,
    _text: string,
    _options?: { dimensions?: number },
  ): Promise<number[]> {
    throw new Error('Amplify does not currently support embedding models')
  }

  private async sendRequest(
    model: ChatModel,
    request: LLMRequest,
    options?: LLMOptions,
  ): Promise<LLMResponseNonStreaming> {
    if (model.providerType !== 'amplify') {
      throw new Error('Model is not an Amplify model')
    }

    const apiKey = this.provider.apiKey?.trim()
    if (!apiKey) {
      throw new Error(
        `Amplify provider ${this.provider.id} has no API key. Add one in Settings.`,
      )
    }

    if (request.tools && request.tools.length > 0) {
      throw new Error(
        'Amplify provider does not currently support Smart Composer tools or MCP calls',
      )
    }

    const messages = request.messages.map((message) =>
      this.toAmplifyMessage(message),
    )

    const latestUserPrompt =
      [...messages].reverse().find((message) => message.role === 'user')
        ?.content ?? ''

    const response = await postJson<AmplifyChatResponse>(
      AMPLIFY_CHAT_URL,
      {
        data: {
          model: request.model,
          temperature: request.temperature ?? 0.7,
          max_tokens: request.max_tokens ?? 4000,
          messages,
          dataSources: [],
          options: {
            skipRag: true,
            model: {
              id: request.model,
            },
            prompt: latestUserPrompt,
          },
        },
      },
      {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          Accept: 'application/json',
        },
        signal: options?.signal,
      },
    )

    if (!response.success) {
      const message = Array.isArray(response.message)
        ? response.message.join('; ')
        : response.message

      throw new Error(
        `Amplify API request failed${message ? `: ${message}` : ''}`,
      )
    }

    if (typeof response.data !== 'string') {
      throw new Error('Amplify returned an unexpected response format')
    }

    return {
      id: uuidv4(),
      created: Math.floor(Date.now() / 1000),
      model: request.model,
      object: 'chat.completion',
      choices: [
        {
          finish_reason: 'stop',
          message: {
            role: 'assistant',
            content: response.data,
          },
        },
      ],
    }
  }

  private toAmplifyMessage(message: RequestMessage): AmplifyMessage {
    if (message.role === 'tool') {
      throw new Error(
        'Amplify provider does not currently support tool-result messages',
      )
    }

    if (typeof message.content === 'string') {
      return {
        role: message.role,
        content: message.content,
      }
    }
    return {
      role: message.role,
      content: message.content
        .map((part) => {
          if (part.type === 'text') {
            return part.text
          }

          throw new Error(
            'Amplify provider currently supports text-only messages',
          )
        })
        .join('\n'),
    }
  }

  private async *singleChunkStream(
    response: LLMResponseNonStreaming,
  ): AsyncIterable<LLMResponseStreaming> {
    const choice = response.choices[0]

    yield {
      id: response.id,
      created: response.created,
      model: response.model,
      object: 'chat.completion.chunk',
      choices: [
        {
          finish_reason: choice?.finish_reason ?? 'stop',
          delta: {
            role: 'assistant',
            content: choice?.message.content ?? '',
          },
        },
      ],
    }
  }
}
