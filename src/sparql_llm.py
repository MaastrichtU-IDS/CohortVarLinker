import os
import rdflib
from typing import Dict, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
# from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_community.graphs.rdf_graph import RdfGraph
from langchain_core.messages import HumanMessage
from langchain_community.chains.graph_qa.prompts import (
    SPARQL_GENERATION_SELECT_PROMPT,
    SPARQL_GENERATION_UPDATE_PROMPT,
    SPARQL_INTENT_PROMPT,
    SPARQL_QA_PROMPT,
)
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_together import ChatTogether

class MultiAgentRDFQA:
    def __init__(self, source_file: str, local_copy: str):
        """
        Initializes the Multi-Agent RDF QA System.
        
        :param source_file: Path to the RDF data source file.
        :param local_copy: Local RDF graph storage.
        """
        self.llm = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                temperature=0,
                api_key="a44c707fe4215fabe2776b064b9544918fdfeb99b568292d3327c3d9a9656d19"
                )

        # Load RDF Graph
        self.rdf_graph = RdfGraph(source_file=source_file, standard="rdf",
                                  local_copy=local_copy, serialization="trig")

        # Initialize LLM Chains for each function
        self.sparql_generation_select_chain = LLMChain(llm=self.llm, prompt=SPARQL_GENERATION_SELECT_PROMPT)
        self.sparql_generation_update_chain = LLMChain(llm=self.llm, prompt=SPARQL_GENERATION_UPDATE_PROMPT)
        self.sparql_intent_chain = LLMChain(llm=self.llm, prompt=SPARQL_INTENT_PROMPT)
        self.qa_chain = LLMChain(llm=self.llm, prompt=SPARQL_QA_PROMPT)
        self.run_manager = CallbackManagerForChainRun.get_noop_manager()

        # Build multi-agent system
        self.graph = self.build_graph()

    class State(MessagesState):
        next: str

    def build_graph(self):
        """Builds the multi-agent system graph."""
        graph_builder = StateGraph(self.State)

        graph_builder.add_node("supervisor", self.supervisor)
        graph_builder.add_node("query_generator", self.query_generator)
        graph_builder.add_node("query_validator", self.query_validator)
        graph_builder.add_node("query_executor", self.query_executor)
        graph_builder.add_node("query_verifier", self.query_verifier)

        graph_builder.add_edge(START, "supervisor")

        return graph_builder.compile()

    def query_generator(self, state: State) -> Command[Literal["query_validator"]]:
        """Agent that generates an initial SPARQL query from a natural language question."""
        user_query = state["messages"][-1].content
        self.run_manager.on_text(f"📝 Generating SPARQL query for: {user_query}", verbose=True)
        callbacks = self.run_manager.get_child()
        # Determine query type (SELECT or UPDATE)
        intent = self.sparql_intent_chain.invoke({"prompt": user_query})["text"].strip()
        # .strip()
        print(f"Intent: {intent}")
        if "SELECT" in intent:
            sparql_generation_chain = self.sparql_generation_select_chain
        elif "UPDATE" in intent:
            sparql_generation_chain = self.sparql_generation_update_chain
        else:
            raise ValueError("Only SELECT and UPDATE queries are supported.")
       
        # Generate SPARQL Query
        generated_sparql = sparql_generation_chain.invoke({"prompt": user_query, "schema": self.rdf_graph.get_schema}, callbacks=callbacks)
        generated_sparql = generated_sparql.get("text").strip()
        print(f"Generated SPARQL: {generated_sparql}")  
        self.run_manager.on_text(f"✅ SPARQL Query Generated:\n{generated_sparql}", verbose=True)
        return Command(
            update={"messages": [HumanMessage(content=generated_sparql, name="query_generator")]},
            goto="query_validator"
        )

    def query_validator(self, state: State) -> Command[Literal["query_executor", "query_generator"]]:
        """Agent that validates the generated SPARQL query for correctness."""
        sparql_query = state["messages"][-1].content
        self.run_manager.on_text(f"🔍 Validating SPARQL query:\n{sparql_query}", verbose=True)
        callbacks = self.run_manager.get_child()
        validation_prompt = f"""
        Validate and correct the following SPARQL query:

        ```sparql
        {sparql_query}
        ```

        Ensure:
        - The query follows proper SPARQL syntax.
        - It uses correct RDF prefixes and relationships.
        - It is a valid `SELECT` or `UPDATE` query.

        If valid, return only "VALID".
        If invalid, return the corrected query.
        """

        validation_response = self.llm.invoke([HumanMessage(content=validation_prompt)], callbacks=callbacks)
        print(f"Validation Response: {validation_response.content}")
        validated_query = validation_response.content.strip().lower()

        if validated_query == "valid":
            print("✅ Query is valid! Proceeding to execution...")
            return Command(goto="query_executor")


        self.run_manager.on_text(f"🔄 Refining query...", verbose=True)
        return Command(
            update={"messages": [HumanMessage(content=validated_query, name="query_validator")]},
            goto="query_generator"
        )

    def query_executor(self, state: State) -> Command[Literal["query_verifier", "query_validator"]]:
        """Agent that executes the validated SPARQL query on the RDF knowledge graph."""
        sparql_query = state["messages"][-1].content
        # print(f"🚀 Executing SPARQL Query:\n{sparql_query}")
        self.run_manager.on_text(f"🚀 Executing SPARQL Query:\n{sparql_query}", verbose=True)
      
        g = rdflib.Graph()
        g.parse(self.rdf_graph.local_copy, format="trig")

        try:
            results = g.query(sparql_query)
            formatted_results = [row.asdict() for row in results]

            if formatted_results:
                print("✅ Query execution successful!")
                return Command(
                    update={"messages": [HumanMessage(content=str(formatted_results), name="query_executor")]},
                    goto="query_verifier"
                )
            else:
                print("⚠️ Query returned no results. Refining...")
                return Command(goto="query_validator")

        except Exception as e:
            # print(f"❌ Query Execution Failed: {e}")
            self.run_manager.on_text(f"❌ Query Execution Failed: {e}", verbose=True)
            return Command(goto="query_validator")

    def query_verifier(self, state: State) -> Command[Literal["query_executor", END]]:
        """Agent that verifies if the query results align with the user's original question."""
        original_query = state["messages"][0].content
        query_results = state["messages"][-1].content

        # print(f"🔍 Verifying results for: {original_query}")
        self.run_manager.on_text(f"🔍 Verifying results for: {original_query}", verbose=True)
        callbacks = self.run_manager.get_child()
        verification_prompt = f"""
        Given the original query: "{original_query}"

        The executed SPARQL query results:

        {query_results}

        Do the results match the user's intent?
        If correct, return "VALID".
        If incorrect, suggest modifications.
        """

        verification_response = self.llm.invoke([HumanMessage(content=verification_prompt)], callbacks=callbacks)
        verification_result = verification_response.content.strip().lower()

        if "valid" in verification_result or "correct" in verification_result:
            print("✅ Results are valid. Process complete!")
            return Command(goto=END)

        # print("🔄 Refining query...")
        self.run_manager.on_text("🔄 Refining query...", verbose=True)
        return Command(goto="query_executor")

    def supervisor(self, state: State) -> Command[Literal["query_generator", "query_validator", "query_executor", "query_verifier", END]]:
        """Supervisor that routes tasks between different agents."""
        last_agent = state["messages"][-1].name

        if last_agent == "user":
            return Command(goto="query_generator")
        elif last_agent == "query_generator":
            return Command(goto="query_validator")
        elif last_agent == "query_validator":
            return Command(goto="query_executor")
        elif last_agent == "query_executor":
            return Command(goto="query_verifier")
        elif last_agent == "query_verifier":
            return Command(goto=END)

        return Command(goto=END)

    def invoke(self, user_query: str):
        """Runs the multi-agent system for RDF QA."""
        for state in self.graph.stream({"messages": [HumanMessage(content=user_query, name="user")]}, {"recursion_limit": 10}):
            print(state)
            print("---")


# Example Usage
if __name__ == "__main__":
    system = MultiAgentRDFQA(
        source_file="/Users/komalgilani/Desktop/chexo_knowledge_graph/data/graphs/studies_metadata.ttl",
        local_copy="/Users/komalgilani/Desktop/chexo_knowledge_graph/data/graphs/qa_retrieval/chexo.ttl"
    )

    user_query = "List all studies conducted before 2020."
    system.invoke(user_query)
