"""Guard the target, time parser and one-person-per-row data contracts."""
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import collapse_locations, parse_hour, target_from_grav


class DataContracts(unittest.TestCase):
    def test_target_keeps_fatalities_and_excludes_slight_injuries(self):
        self.assertEqual(target_from_grav(pd.Series([1, 2, 3, 4])).tolist(), [0, 1, 1, 0])
        with self.assertRaises(ValueError):
            target_from_grav(pd.Series([-1, 2]))

    def test_hour_formats_and_invalid_times(self):
        result = parse_hour(pd.Series(["07:32", "1325", "5", "00:50", "24:00", "12:78", None]))
        self.assertEqual(result[:4].tolist(), [7., 13., 0., 0.])
        self.assertTrue(result[4:].isna().all())

    def test_conflicting_roads_do_not_duplicate_people(self):
        roads = pd.DataFrame({"Num_Acc": ["a", "a", "b"], "surf": ["1", "2", "1"],
                              "catr": ["3", "3", "2"], "vma": ["50", "50", "80"]})
        collapsed, audit = collapse_locations(roads)
        people = pd.DataFrame({"Num_Acc": ["a", "a", "b"], "person": [1, 2, 3]})
        merged = people.merge(collapsed, on="Num_Acc", validate="many_to_one")
        self.assertEqual(len(merged), 3)
        self.assertTrue(merged.loc[merged.Num_Acc.eq("a"), "surf"].isna().all())
        self.assertEqual(merged.loc[merged.Num_Acc.eq("a"), "vma"].tolist(), ["50", "50"])
        self.assertEqual(audit["conflicting_location_attributes"]["surf"], 1)


if __name__ == "__main__":
    unittest.main()
