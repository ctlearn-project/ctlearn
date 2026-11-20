"""Render CWL descriptions, interfaces, and workflow graphs.

Inspired by datapipe's docs/_extensions/cwl_workflow.py:
https://gitlab.cta-observatory.org/cta-computing/dpps/datapipe/datapipe
Copyright (c) 2022, Cherenkov Telescope Array Observatory Consortium
(BSD 3-Clause License).
"""

import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.ext.graphviz import graphviz


def describe_type(value):
    """Render the common CWL input and output types."""
    if isinstance(value, list):
        return " | ".join(describe_type(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "enum":
            return "enum: " + ", ".join(value["symbols"])
        if value.get("type") == "array":
            return describe_type(value["items"]) + "[]"
        return str(value.get("type", "unknown"))
    return str(value)


class CWLWorkflowDirective(Directive):
    """Document a CWL file using its label, documentation and interface."""

    required_arguments = 1
    has_content = False
    option_spec = {"package": directives.unchanged_required}

    def run(self):
        env = self.state.document.settings.env
        argument = self.arguments[0]
        package = self.options.get("package")
        if package and (Path(argument).is_absolute() or ".." in Path(argument).parts):
            raise self.error("CWL package resource must be a relative path")
        location = f"{package}:{argument}" if package else argument
        try:
            if package:
                resource = files(package).joinpath(*argument.split("/"))
            else:
                document = Path(env.doc2path(env.docname)).resolve()
                resource = (document.parent / argument).resolve()
            env.note_dependency(str(resource))
            with resource.open(encoding="utf-8") as stream:
                tool = yaml.safe_load(stream)
        except (ImportError, OSError, TypeError, UnicodeError, yaml.YAMLError) as exc:
            raise self.error(f"Cannot read CWL file {location}: {exc}") from exc
        if not isinstance(tool, dict):
            raise self.error(f"CWL file {location} must contain a mapping")

        name = resource.name
        stem = Path(name).stem
        section = nodes.section(ids=[nodes.make_id(stem)])
        section += nodes.title(text=tool.get("label", stem))
        section += nodes.paragraph(text=f"{tool.get('class', 'CWL')}: {name}")
        self.add_description(section, tool.get("doc", ""))
        for heading, key in (("Inputs", "inputs"), ("Outputs", "outputs")):
            section += nodes.rubric(text=heading)
            section += self.make_table(tool.get(key, {}))
        return [section]

    def add_description(self, parent, text):
        if text:
            self.state.nested_parse(
                StringList(text.splitlines()), self.content_offset, parent
            )

    def make_table(self, fields):
        table = nodes.table()
        group = nodes.tgroup(cols=3)
        table += group
        for width in (2, 2, 6):
            group += nodes.colspec(colwidth=width)
        head = nodes.thead()
        body = nodes.tbody()
        group += head
        group += body
        header = nodes.row()
        for title in ("ID", "Type", "Description"):
            entry = nodes.entry()
            entry += nodes.paragraph(text=title)
            header += entry
        head += header
        if isinstance(fields, list):
            fields = {item["id"]: item for item in fields}
        for name, spec in fields.items():
            spec = spec if isinstance(spec, dict) else {"type": spec}
            row = nodes.row()
            for value in (name, describe_type(spec.get("type", "unknown"))):
                entry = nodes.entry()
                paragraph = nodes.paragraph()
                paragraph += nodes.literal(text=value)
                entry += paragraph
                row += entry
            description = nodes.entry()
            self.add_description(description, spec.get("doc", ""))
            row += description
            body += row
        return table


class CWLWorkflowGraphDirective(Directive):
    """Render a CWL Workflow's data-flow graph."""

    required_arguments = 1
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        document = Path(env.doc2path(env.docname)).resolve()
        path = (document.parent / self.arguments[0]).resolve()
        env.note_dependency(str(path))

        dot = env.app.config.graphviz_dot
        if shutil.which(dot) is None:
            raise self.error(f"Graphviz executable not found: {dot}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "cwltool", "--print-dot", str(path)],
                capture_output=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise self.error(
                f"Cannot generate CWL workflow graph for {path}:\n{exc.stderr.strip()}"
            ) from exc
        if not result.stdout.strip():
            raise self.error(f"cwltool produced an empty workflow graph for {path}")

        graph = graphviz()
        graph["code"] = result.stdout
        graph["options"] = {"docname": env.docname}
        graph["alt"] = f"CWL workflow graph for {path.name}"
        return [graph]


def setup(app):
    app.setup_extension("sphinx.ext.graphviz")
    app.config.graphviz_output_format = "svg"
    app.add_directive("cwl_workflow", CWLWorkflowDirective)
    app.add_directive("cwl_workflow_graph", CWLWorkflowGraphDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
