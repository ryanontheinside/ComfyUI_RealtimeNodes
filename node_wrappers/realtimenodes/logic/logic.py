from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.utils.general import AlwaysEqualProxy


class LazyCondition(ControlNodeBase):
    DESCRIPTION = (
        "Uses lazy evaluation to truly skip execution of unused paths. Maintains state of the last value to circumvent feedback loops."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "condition": (
                    AlwaysEqualProxy("*"),
                    {
                        "tooltip": "When truthy (non-zero, non-empty, non-None), evaluates and returns if_true path. When falsy, returns either fallback or previous state of if_true.",
                        "forceInput": True,
                    },
                ),
                "if_true": (
                    AlwaysEqualProxy("*"),
                    {"lazy": True, "tooltip": "The path that should only be evaluated when condition is truthy"},
                ),
                "fallback": (
                    AlwaysEqualProxy("*"),
                    {"tooltip": "Alternative value to use when condition is falsy or no previous state of if_true"},
                ),
                "use_fallback": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When False, uses last successful if_true result (if one exists). When True, uses fallback value",
                    },
                ),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/utility"

    def check_lazy_status(self, condition, if_true, fallback, use_fallback):
        """Only evaluate the if_true path if condition is truthy."""
        needed = ["fallback"]  # Always need the fallback value
        if condition:
            needed.append("if_true")
        return needed

    def update(self, condition, if_true, fallback, use_fallback):
        """Route to either if_true output or fallback value, maintaining state of last if_true."""
        state = self.get_state({"prev_output": None})

        if condition:  # Let Python handle truthiness
            # Update last state when we run if_true path
            state["prev_output"] = if_true if if_true is not None else fallback
            if hasattr(if_true, "detach"):
                state["prev_output"] = if_true.detach().clone()
            self.set_state(state)
            return (if_true,)
        else:
            if use_fallback or state["prev_output"] is None:
                return (fallback,)
            else:
                return (state["prev_output"],)


class LogicOperator(ControlNodeBase):
    DESCRIPTION = "Performs logical operations (AND, OR, NOT, XOR) on inputs based on their truthiness"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation": (
                    ["AND", "OR", "NOT", "XOR"],
                    {"default": "AND", "tooltip": "Logical operation to perform"},
                ),
                "input_a": (
                    AlwaysEqualProxy("*"),
                    {
                        "tooltip": "First input to evaluate for truthiness",
                        "forceInput": True,
                    },
                ),
                "always_execute": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
            "optional": {
                "input_b": (
                    AlwaysEqualProxy("*"),
                    {
                        "tooltip": "Second input to evaluate for truthiness (not used for NOT operation)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/logic"

    def update(self, operation, input_a, always_execute=True, input_b=None):
        """Perform the selected logical operation on inputs based on their truthiness."""
        a = bool(input_a)

        if operation == "NOT":
            return (not a,)

        # For all other operations, we need input_b
        b = bool(input_b)

        if operation == "AND":
            return (a and b,)
        elif operation == "OR":
            return (a or b,)
        elif operation == "XOR":
            return (a != b,)

        # Should never get here, but just in case
        return (False,)
