import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from qahs import arc_cli
from qahs.arc_reasoner import ARC_PROPOSAL_SCHEMA, ARC_REFLECTION_SCHEMA, ARC_REVIEW_SCHEMA, OpenAIArcReasoner
from qahs.compositional import SUCCESS_AUDIT_SCHEMA, VOCABULARY_SCHEMA
from qahs.information_gain import HYPOTHESIS_SCHEMA, InformationGainReasoner
from qahs.providers.anthropic_provider import UNSUPPORTED_KEYWORDS, AnthropicJSONCaller, strip_unsupported
from qahs.structured_world_models import StructuredWorldReasoner


def response(text='{"ok": true}', stop_reason="end_turn", category=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        usage=SimpleNamespace(input_tokens=120, output_tokens=40, cache_read_input_tokens=0),
    )


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)

    def stream(self, **kwargs):
        client = self

        class Stream:
            def __enter__(self):
                client.requests.append(kwargs)
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return client.responses.pop(0)

        return Stream()


def harness_schemas():
    caller = SimpleNamespace(model="m", reasoning_effort="medium")
    return {
        "proposal": ARC_PROPOSAL_SCHEMA, "reflection": ARC_REFLECTION_SCHEMA, "review": ARC_REVIEW_SCHEMA,
        "vocabulary": VOCABULARY_SCHEMA, "success_audit": SUCCESS_AUDIT_SCHEMA,
        "v11_hypotheses": InformationGainReasoner(caller).hypothesis_schema,
        "v12_hypotheses": StructuredWorldReasoner(caller).hypothesis_schema,
    }


def walk(schema):
    if isinstance(schema, dict):
        yield schema
        for value in schema.values():
            yield from walk(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from walk(value)


class AnthropicCallerTests(unittest.TestCase):
    def test_schema_effort_and_usage_are_sent_and_recorded(self):
        client = FakeClient(response())
        caller = AnthropicJSONCaller(model="claude-opus-5-5", reasoning_effort="medium", client=client)
        self.assertEqual(caller.call_json("prompt", ARC_REFLECTION_SCHEMA, "arc_qahs_reflection"), {"ok": True})
        request = client.requests[0]
        self.assertEqual(request["model"], "claude-opus-5-5")
        self.assertEqual(request["output_config"]["format"], {"type": "json_schema", "schema": ARC_REFLECTION_SCHEMA})
        self.assertEqual(request["output_config"]["effort"], "medium")
        self.assertEqual(request["messages"], [{"role": "user", "content": "prompt"}])
        self.assertNotIn("temperature", request)
        self.assertNotIn("fallbacks", request)
        self.assertEqual(caller.last_usage["input_tokens"], 120)
        self.assertEqual(caller.settings["provider"], "anthropic")

    def test_usage_totals_and_call_errors_reach_settings(self):
        caller = AnthropicJSONCaller(client=FakeClient(response(), response(), response(stop_reason="refusal", category="cyber")))
        caller.call_json("prompt", {})
        caller.call_json("prompt", {})
        with self.assertRaises(RuntimeError):
            caller.call_json("prompt", {})
        settings = caller.settings
        self.assertEqual(settings["request_count"], 3)
        self.assertEqual(settings["usage_totals"]["input"], 3 * caller.last_usage["input_tokens"])
        self.assertEqual(len(settings["call_errors"]), 1)
        self.assertIn("refused", settings["call_errors"][0])
        self.assertIn("cyber", settings["call_errors"][0])

    def test_effort_none_is_rejected(self):
        with self.assertRaises(RuntimeError):
            AnthropicJSONCaller(reasoning_effort="none", client=FakeClient())

    def test_refusal_and_truncation_raise(self):
        for bad in (response(stop_reason="refusal", category="cyber"), response(text='{"ok"', stop_reason="max_tokens")):
            caller = AnthropicJSONCaller(client=FakeClient(bad))
            with self.assertRaises(RuntimeError):
                caller.call_json("prompt", {})

    def test_v7_reasoner_retries_a_refusal(self):
        client = FakeClient(response(stop_reason="refusal"), response('{"value": 3}'))
        reasoner = OpenAIArcReasoner(AnthropicJSONCaller(client=client), retry_delay=0)
        self.assertEqual(reasoner._call_json("p", {}, "n"), {"value": 3})
        self.assertEqual(reasoner.last_call_count, 2)

    def test_images_become_base64_blocks_before_the_prompt(self):
        client = FakeClient(response())
        caller = AnthropicJSONCaller(client=client)
        caller.call_json("prompt", {}, "n", ["data:image/png;base64,AAAA"])  # positional, as v10/v11 pass it
        content = client.requests[0]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}})
        self.assertEqual(content[-1], {"type": "text", "text": "prompt"})


class SchemaCompatibilityTests(unittest.TestCase):
    def test_every_harness_schema_compiles_under_anthropic_limits(self):
        for name, schema in harness_schemas().items():
            with self.subTest(schema=name):
                sent = strip_unsupported(schema)
                optional = unions = 0
                for node in walk(sent):
                    self.assertFalse(set(UNSUPPORTED_KEYWORDS) & set(node), node)
                    self.assertLessEqual(node.get("minItems", 0), 1)
                    if node.get("type") == "object":
                        self.assertIs(node.get("additionalProperties"), False)
                        optional += len(set(node.get("properties", {})) - set(node.get("required", [])))
                    unions += "anyOf" in node or isinstance(node.get("type"), list)
                self.assertLessEqual(optional, 24)
                self.assertLessEqual(unions, 16)

    def test_stripped_constraints_are_restated_and_original_is_untouched(self):
        sent = strip_unsupported(HYPOTHESIS_SCHEMA)
        self.assertIn("minItems=6", sent["properties"]["hypotheses"]["description"])
        self.assertIn("maxItems=10", sent["properties"]["hypotheses"]["description"])
        self.assertEqual(HYPOTHESIS_SCHEMA["properties"]["hypotheses"]["maxItems"], 10)

    def test_constraint_violation_is_retried_then_arrays_are_truncated(self):
        schema = {"type": "object", "additionalProperties": False, "required": ["xs"],
                  "properties": {"xs": {"type": "array", "maxItems": 2, "items": {"type": "integer"}}}}
        long, short = json.dumps({"xs": [1, 2, 3]}), json.dumps({"xs": [1]})
        caller = AnthropicJSONCaller(client=FakeClient(response(long), response(short)))
        self.assertEqual(caller.call_json("p", schema), {"xs": [1]})
        self.assertEqual((caller.constraint_retries, caller.constraint_truncations), (1, 0))
        caller = AnthropicJSONCaller(client=FakeClient(*(response(long) for _ in range(3))))
        self.assertEqual(caller.call_json("p", schema), {"xs": [1, 2]})
        self.assertEqual((caller.constraint_retries, caller.constraint_truncations), (2, 1))


class ProviderCliTests(unittest.TestCase):
    def run_cli(self, harness):
        captured = {}
        episode = SimpleNamespace(success=False, turns=[], error=None)
        trace = SimpleNamespace(without_qahs=episode, with_classical=episode, with_qahs=episode, model_settings={})

        def runner(environment_factory, reasoner, *args):
            captured["caller"] = reasoner.caller
            return trace

        def v7_runner(environment_factory, reasoner_factory, *args, **kwargs):
            captured["caller"] = reasoner_factory().caller
            return trace

        argv = ["qahs-arc", "--source", "arc", "--provider", "anthropic", "--harness", harness, "--condition", "compare"]
        patches = [
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}),
            mock.patch.object(arc_cli, "run_structured_world_comparison", runner),
            mock.patch.object(arc_cli, "run_information_gain_comparison", runner),
            mock.patch.object(arc_cli, "run_compositional_comparison", runner),
            mock.patch.object(arc_cli, "run_arc_comparison", v7_runner),
            mock.patch.object(arc_cli, "save_arc_trace", lambda trace, directory: "trace.json"),
            mock.patch("builtins.print"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        arc_cli.main()
        return captured["caller"], trace

    def test_anthropic_provider_reaches_every_harness(self):
        for harness in ("v7", "v10", "v11", "v12"):
            with self.subTest(harness=harness):
                caller, trace = self.run_cli(harness)
                self.assertIsInstance(caller, AnthropicJSONCaller)
                self.assertEqual(caller.model, os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"))
                if harness != "v7":
                    self.assertEqual(trace.model_settings["provider"], "anthropic")


if __name__ == "__main__":
    unittest.main()
