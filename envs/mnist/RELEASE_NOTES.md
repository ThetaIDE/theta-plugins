# 1.2.0

Compatibility is declared once, as capability `requires` rules in the manifest. The package-level `paradigms` list, profile `compatible_paradigms`, `execution_features.input_types` and class-level `get_supported_paradigms` are gone; nothing reads them. Requires an IDE with Theta SDK 3.1 (the compatibility-requirements release); earlier IDEs reject the manifest.
