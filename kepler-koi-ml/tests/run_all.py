import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

tests = [
    ("tests.test_features", "test_bls_on_synthetic"),
    ("tests.test_plots", "test_plot_labels"),
    ("tests.test_training", "test_training_artifacts"),
    ("tests.test_koi_utils", "test_koi_row_for_target_mask_alignment"),
    ("tests.test_koi_utils", "test_fetch_uses_cache"),
    ("tests.test_fetchers_mocked", "test_fetch_koi_mock"),
    ("tests.test_fetchers_mocked", "test_fetch_toi_mock"),
    ("tests.test_fetchers_mocked", "test_fetch_k2_mock"),
    ("tests.test_resolver", "test_resolve_features_for_target_offline"),
]

failed = False
for mod_name, func_name in tests:
    try:
        mod = importlib.import_module(mod_name)
        getattr(mod, func_name)()
        print(f"✅ {mod_name}.{func_name} passed")
    except Exception as e:
        failed = True
        print(f"❌ {mod_name}.{func_name} failed: {e}")

if failed:
    sys.exit(1)
else:
    print("🎉 all tests passed")
