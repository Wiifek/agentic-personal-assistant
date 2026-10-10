import { ChatGroq } from "@langchain/groq";
import { createAgent } from "langchain";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { searchKnowledgeBase } from "./tools.js";

// Create a memory saver for persisting conversation history
const checkpointer = new MemorySaver();

export async function runAgent({ sessionId = "default", message }) {
  try {
    const model = new ChatGroq({
      model: "openai/gpt-oss-120b",
      temperature: 0,
    });

    const agent = createAgent({
      model,
      tools: [searchKnowledgeBase],
      checkpointer,
      systemPrompt:
        `You are a helpful AI assistant with access to a knowledge base. When users ask questions,
         search the knowledge base using the available tools to find relevant information. Be concise and accurate.`,
    });

    console.log(`🤖 Running agent for: "${message}"`);

    // Invoke here has an agentic behavior and it will decide to use the tool or not.
    let response;
    try {
      console.log("Starting agent.invoke()");

      response = await agent.invoke(
        {
          messages: [{ role: "user", content: message }],
        },
        {
          configurable: {
            thread_id: sessionId, // This maintains conversation history per session
          },
        }
      );
      console.log("agent.invoke() completed successfully");
    } catch (error) {
      console.error("agent.invoke() failed:", error);
      console.error("Error cause:", error.cause);
      console.error("Stack:", error.stack);
      throw error;
    }

    // Extract the last message content
    const lastMessage = response.messages[response.messages.length - 1];
    const output = lastMessage?.content || "";

    console.log(`✅ Agent response: ${output.slice(0, 100)}...`);

    return { output };
  } catch (error) {
    console.error("❌ Error in runAgent:", error);
    throw error;
  }
}
