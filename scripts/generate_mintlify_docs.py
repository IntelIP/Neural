#!/usr/bin/env python3
"""
Generate Mintlify documentation from Neural SDK code.

This script extracts docstrings, type hints, and function signatures
to automatically generate MDX documentation files.
"""

import ast
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add neural_sdk to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk import NeuralSDK
from neural_sdk.streaming import NeuralWebSocket
from neural_sdk.backtesting import BacktestEngine, BacktestConfig


class DocstringExtractor(ast.NodeVisitor):
    """Extract docstrings and metadata from Python AST."""
    
    def __init__(self):
        self.classes = {}
        self.functions = {}
        self.current_class = None
    
    def visit_ClassDef(self, node):
        """Extract class documentation."""
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'methods': {},
            'bases': [base.id for base in node.bases if isinstance(base, ast.Name)]
        }
        
        self.current_class = node.name
        self.classes[node.name] = class_info
        self.generic_visit(node)
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        """Extract function/method documentation."""
        func_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'args': [],
            'returns': None,
            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        }
        
        # Extract arguments
        for arg in node.args.args:
            if arg.arg != 'self':
                arg_info = {'name': arg.arg, 'type': None}
                if arg.annotation:
                    arg_info['type'] = ast.unparse(arg.annotation)
                func_info['args'].append(arg_info)
        
        # Extract return type
        if node.returns:
            func_info['returns'] = ast.unparse(node.returns)
        
        # Store in appropriate location
        if self.current_class:
            self.classes[self.current_class]['methods'][node.name] = func_info
        else:
            self.functions[node.name] = func_info
        
        self.generic_visit(node)


def extract_module_docs(module_path: Path) -> Dict[str, Any]:
    """Extract documentation from a Python module."""
    with open(module_path, 'r') as f:
        tree = ast.parse(f.read())
    
    extractor = DocstringExtractor()
    extractor.visit(tree)
    
    return {
        'module': module_path.stem,
        'docstring': ast.get_docstring(tree),
        'classes': extractor.classes,
        'functions': extractor.functions
    }


def generate_class_mdx(class_name: str, class_info: Dict, module_name: str) -> str:
    """Generate MDX documentation for a class."""
    mdx = f"""---
title: '{class_name}'
description: '{class_info.get('docstring', '').split('.')[0] if class_info.get('docstring') else f'{class_name} class'}'
---

## Overview

{class_info.get('docstring', f'Documentation for {class_name} class.')}

"""
    
    # Add constructor if present
    if '__init__' in class_info['methods']:
        init_method = class_info['methods']['__init__']
        mdx += "## Constructor\n\n"
        
        for arg in init_method.get('args', []):
            mdx += f"""<ParamField path="{arg['name']}" type="{arg.get('type', 'Any')}" required>
  {arg['name']} parameter
</ParamField>

"""
        
        mdx += """<CodeGroup>

```python Python
from neural_sdk.{module} import {class_name}

# Initialize {class_name}
instance = {class_name}()
```

</CodeGroup>

""".format(module=module_name, class_name=class_name)
    
    # Add methods
    mdx += "## Methods\n\n"
    
    for method_name, method_info in class_info['methods'].items():
        if method_name.startswith('_') and method_name != '__init__':
            continue
        
        mdx += f"""### `{method_name}({', '.join(arg['name'] for arg in method_info.get('args', []))})`

{method_info.get('docstring', f'{method_name} method.')}

"""
        
        # Add parameters
        for arg in method_info.get('args', []):
            mdx += f"""<ParamField path="{arg['name']}" type="{arg.get('type', 'Any')}">
  {arg['name']} parameter
</ParamField>

"""
        
        # Add return type
        if method_info.get('returns'):
            mdx += f"""<ResponseField name="result" type="{method_info['returns']}">
  Method result
</ResponseField>

"""
        
        mdx += "---\n\n"
    
    return mdx


def generate_api_reference():
    """Generate API reference documentation for Neural SDK."""
    docs_dir = Path(__file__).parent.parent / 'docs-mintlify' / 'api-reference'
    
    # Core modules to document
    modules = [
        ('neural_sdk/core/client.py', 'sdk'),
        ('neural_sdk/streaming/websocket.py', 'streaming'),
        ('neural_sdk/backtesting/engine.py', 'backtesting'),
        ('neural_sdk/trading/risk_manager.py', 'trading'),
    ]
    
    for module_path, category in modules:
        full_path = Path(__file__).parent.parent / module_path
        
        if not full_path.exists():
            print(f"Skipping {module_path} - file not found")
            continue
        
        # Extract documentation
        docs = extract_module_docs(full_path)
        
        # Create category directory
        category_dir = docs_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate MDX for each class
        for class_name, class_info in docs['classes'].items():
            mdx_content = generate_class_mdx(class_name, class_info, docs['module'])
            
            # Save MDX file
            output_file = category_dir / f"{class_name.lower()}.mdx"
            output_file.write_text(mdx_content)
            print(f"Generated {output_file}")


def generate_openapi_spec():
    """Generate OpenAPI specification from SDK methods."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Neural SDK API",
            "version": "1.1.0",
            "description": "API specification for Neural SDK"
        },
        "servers": [
            {
                "url": "https://api.kalshi.com/v1",
                "description": "Kalshi API"
            }
        ],
        "paths": {},
        "components": {
            "schemas": {},
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        },
        "security": [
            {"BearerAuth": []}
        ]
    }
    
    # Extract methods from SDK classes
    for cls in [NeuralSDK, NeuralWebSocket, BacktestEngine]:
        for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
            if name.startswith('_'):
                continue
            
            # Generate path
            path = f"/{cls.__name__}/{name}"
            
            # Extract parameters from signature
            sig = inspect.signature(method) if hasattr(method, '__func__') else None
            params = []
            
            if sig:
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'cls']:
                        params.append({
                            "name": param_name,
                            "in": "query",
                            "required": param.default == inspect.Parameter.empty,
                            "schema": {"type": "string"}
                        })
            
            # Add to spec
            spec["paths"][path] = {
                "post": {
                    "summary": method.__doc__.split('\n')[0] if method.__doc__ else name,
                    "operationId": f"{cls.__name__}_{name}",
                    "parameters": params,
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            }
    
    # Save OpenAPI spec
    output_file = Path(__file__).parent.parent / 'docs-mintlify' / 'openapi.json'
    output_file.write_text(json.dumps(spec, indent=2))
    print(f"Generated OpenAPI spec: {output_file}")


def main():
    """Main entry point."""
    print("🚀 Generating Mintlify documentation...")
    
    # Generate API reference
    print("\n📚 Generating API reference...")
    generate_api_reference()
    
    # Generate OpenAPI spec
    print("\n📋 Generating OpenAPI specification...")
    generate_openapi_spec()
    
    print("\n✅ Documentation generation complete!")
    print("\nTo preview the docs:")
    print("  cd docs-mintlify")
    print("  npx mintlify dev")


if __name__ == "__main__":
    main()