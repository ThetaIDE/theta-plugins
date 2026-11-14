"""Open the selected rules resource using the shared workspace resolver."""
from src.app.api.rulesets import rules_folder_targets


def rules_edit_targets(*, value, context):
    return rules_folder_targets(value=value, context=context)
