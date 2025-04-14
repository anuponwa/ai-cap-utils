# Change log

## 0.2.0
* Make system prompt mandatory for `BaseAgent`
* Add `agent_name`, if not provided, generate a random one
* Add `BaseChatInterface` that supplies the base methods, i.t., `chat_interact()`, `run_batch` and `chat_each_message()`
* Add multiple agents support
* Use `BaseMultiAgentLLMRouter` class to route different intents to different agents (**not yet support shared memory**)

## 0.1.0
* First dev version
* Provide `BaseAgent`