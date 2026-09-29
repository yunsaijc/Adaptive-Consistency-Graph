from acg.runtime.tools import ToolSpec, ToolTask, string_parameters


def make_task(max_steps=4):
    calls = []
    task = ToolTask("test", "Read then submit.", [], "submit", max_steps=max_steps)

    def read(args):
        calls.append(args["name"])
        return "value"

    def submit(args):
        task.finish(args["answer"])
        return "submitted"

    task.tools = [
        ToolSpec("read", "Read a value", string_parameters(name="Name"), read),
        ToolSpec(
            "submit", "Submit an answer", string_parameters(answer="Answer"), submit
        ),
    ]
    return task, calls


def test_invalid_batch_has_no_partial_side_effects():
    task, calls = make_task()
    task.execute_batch([("read", {"name": "first"}), ("read", {})])
    assert calls == []
    assert task.steps == 1
    assert task.events[0].name == "invalid_action"


def test_valid_native_batch_executes_in_order():
    task, calls = make_task()
    outputs = task.execute_batch(
        [
            ("read", {"name": "first"}),
            ("read", {"name": "second"}),
            ("submit", {"answer": "answer"}),
        ]
    )
    assert len(outputs) == 3
    assert calls == ["first", "second"]
    assert task.steps == 3 and task.done


def test_submission_must_be_last():
    task, calls = make_task()
    task.execute_batch([("submit", {"answer": "answer"}), ("read", {"name": "first"})])
    assert not task.done and not calls


def test_over_budget_batch_is_rejected_before_execution():
    task, calls = make_task(max_steps=1)
    task.execute_batch([("read", {"name": "first"}), ("read", {"name": "second"})])
    assert not calls and task.exhausted
