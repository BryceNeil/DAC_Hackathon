"""
LLM Interface for ASU Tapeout Agent
===================================

Provides unified interface for multiple LLM providers with LangChain integration.
"""

import os
from typing import Optional, Dict, Any, List, Union
from abc import ABC, abstractmethod

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_community.llms import Ollama  # For local models

# Import for structured output
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class LLMProvider(str, ABC):
    """Enum for LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class RTLCodeOutput(BaseModel):
    """Structured output for RTL generation"""
    rtl_code: str = Field(description="Generated Verilog/SystemVerilog RTL code")
    module_name: str = Field(description="Name of the module")
    comments: str = Field(description="Any comments or explanations about the design")
    warnings: List[str] = Field(default=[], description="Any warnings or notes")


class SDCConstraintsOutput(BaseModel):
    """Structured output for SDC generation"""
    sdc_content: str = Field(description="Generated SDC constraints")
    clock_period: float = Field(description="Clock period in nanoseconds")
    comments: str = Field(description="Explanation of constraints")


class LLMInterface:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self, provider: str = LLMProvider.OPENAI, model: Optional[str] = None):
        """Initialize LLM interface
        
        Args:
            provider: LLM provider to use
            model: Specific model name (optional)
        """
        self.provider = provider
        self.model = model
        self.llm = self._initialize_llm()
        
        # Standard prompt templates
        self.rtl_prompt_template = self._create_rtl_prompt_template()
        self.sdc_prompt_template = self._create_sdc_prompt_template()
        
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the appropriate LLM based on provider
        
        Returns:
            Initialized LLM instance
        """
        if self.provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or API_KEY environment variable not set")
            
            model_name = self.model or "gpt-4"
            return ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key
            )
            
        elif self.provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY or API_KEY environment variable not set")
                
            model_name = self.model or "claude-3-sonnet-20240229"
            return ChatAnthropic(
                model=model_name,
                temperature=0,
                api_key=api_key
            )
            
        elif self.provider == LLMProvider.LOCAL:
            # Example for local models using Ollama
            # model_name = self.model or "codellama"
            # return Ollama(model=model_name, temperature=0)
            raise NotImplementedError("Local model support not yet implemented")
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _create_rtl_prompt_template(self) -> ChatPromptTemplate:
        """Create standard RTL generation prompt template
        
        Returns:
            ChatPromptTemplate for RTL generation
        """
        system_message = """You are an expert RTL designer specializing in Verilog/SystemVerilog.
Your task is to generate synthesizable RTL code based on specifications.

Guidelines:
1. Generate clean, synthesizable Verilog code
2. Use proper coding style with consistent indentation
3. Include appropriate comments
4. Follow best practices for the target technology
5. Ensure the module signature matches exactly what's specified
6. For sequential designs, use proper reset logic
7. For combinational designs, avoid latches"""

        human_template = """Generate RTL code for the following specification:

Problem Name: {problem_name}
Description: {description}
Module Signature: {module_signature}
Clock Period: {clock_period}

Additional Requirements:
{additional_requirements}

Please generate complete, synthesizable RTL code."""

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            HumanMessage(content=human_template)
        ])
    
    def _create_sdc_prompt_template(self) -> ChatPromptTemplate:
        """Create standard SDC generation prompt template
        
        Returns:
            ChatPromptTemplate for SDC generation
        """
        system_message = """You are an expert in timing constraints and SDC (Synopsys Design Constraints).
Your task is to generate appropriate SDC constraints for digital designs.

Guidelines:
1. Create clean, well-commented SDC constraints
2. Include clock definitions with proper periods
3. Set appropriate input/output delays
4. Consider the design type (combinational vs sequential)
5. Use technology-appropriate values (Sky130 PDK)
6. Include false paths where appropriate"""

        human_template = """Generate SDC constraints for the following design:

Problem Name: {problem_name}
Description: {description}
Clock Period: {clock_period}
Module Ports: {ports}
Design Type: {design_type}

Please generate complete SDC constraints."""

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            HumanMessage(content=human_template)
        ])
    
    async def generate_rtl(self, specification: Dict[str, Any]) -> RTLCodeOutput:
        """Generate RTL code using LLM
        
        Args:
            specification: Problem specification dictionary
            
        Returns:
            Structured RTL output
        """
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=RTLCodeOutput)
        
        # Format prompt
        prompt = self.rtl_prompt_template.format(
            problem_name=specification.get('name', 'unknown'),
            description=specification.get('description', ''),
            module_signature=specification.get('module_signature', ''),
            clock_period=specification.get('clock_period', '1.0ns'),
            additional_requirements=specification.get('additional_requirements', 'None')
        )
        
        # Add format instructions
        prompt += f"\n\n{parser.get_format_instructions()}"
        
        # Generate response
        response = await self.llm.ainvoke(prompt)
        
        # Parse structured output
        try:
            return parser.parse(response.content)
        except Exception as e:
            # Fallback to basic parsing
            return RTLCodeOutput(
                rtl_code=response.content,
                module_name=specification.get('name', 'unknown'),
                comments="Generated with fallback parsing",
                warnings=[f"Structured parsing failed: {str(e)}"]
            )
    
    async def generate_sdc(self, design_info: Dict[str, Any]) -> SDCConstraintsOutput:
        """Generate SDC constraints using LLM
        
        Args:
            design_info: Design information dictionary
            
        Returns:
            Structured SDC output
        """
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=SDCConstraintsOutput)
        
        # Format prompt
        prompt = self.sdc_prompt_template.format(
            problem_name=design_info.get('name', 'unknown'),
            description=design_info.get('description', ''),
            clock_period=design_info.get('clock_period', '1.0ns'),
            ports=design_info.get('io_ports', {}),
            design_type=design_info.get('problem_type', 'unknown')
        )
        
        # Add format instructions
        prompt += f"\n\n{parser.get_format_instructions()}"
        
        # Generate response
        response = await self.llm.ainvoke(prompt)
        
        # Parse structured output
        try:
            return parser.parse(response.content)
        except Exception as e:
            # Fallback to basic parsing
            clock_str = design_info.get('clock_period', '1.0ns')
            clock_val = float(clock_str.replace('ns', ''))
            
            return SDCConstraintsOutput(
                sdc_content=response.content,
                clock_period=clock_val,
                comments="Generated with fallback parsing"
            )
    
    def create_custom_chain(self, prompt_template: ChatPromptTemplate) -> Any:
        """Create a custom LangChain for specific tasks
        
        Args:
            prompt_template: Custom prompt template
            
        Returns:
            LangChain instance
        """
        return prompt_template | self.llm
    
    def switch_provider(self, provider: str, model: Optional[str] = None):
        """Switch to a different LLM provider
        
        Args:
            provider: New provider to use
            model: Specific model name (optional)
        """
        self.provider = provider
        self.model = model
        self.llm = self._initialize_llm()
    
    async def analyze_error(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze an error and suggest fixes
        
        Args:
            error_message: The error message
            context: Additional context about the error
            
        Returns:
            Suggested fix or explanation
        """
        prompt = f"""Analyze this error in the context of RTL design and suggest a fix:

Error: {error_message}

Context:
- Design Type: {context.get('design_type', 'unknown')}
- Module: {context.get('module_name', 'unknown')}
- Stage: {context.get('stage', 'unknown')}

Provide a concise explanation and suggested fix."""

        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def refine_rtl(self, original_rtl: str, feedback: str) -> str:
        """Refine RTL code based on feedback
        
        Args:
            original_rtl: Original RTL code
            feedback: Feedback or error messages
            
        Returns:
            Refined RTL code
        """
        prompt = f"""Refine this RTL code based on the feedback:

Original RTL:
```verilog
{original_rtl}
```

Feedback:
{feedback}

Generate the corrected RTL code:"""

        response = await self.llm.ainvoke(prompt)
        return response.content 