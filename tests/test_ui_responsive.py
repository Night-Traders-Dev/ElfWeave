import unittest
from rich.cells import cell_len

from src.common.ui import Step
from src.modules.knowledge_agent import format_matches
from src.modules.weather import build_weather_harness_output
from src.modules.weather_logic import DayForecast, HourlySlot, MemorySnapshot, WeatherComparison, WeatherReport


class ResponsiveUiTests(unittest.TestCase):
    def test_step_render_stays_within_requested_width(self) -> None:
        step = Step(name="a very long step name that should be clipped").done(
            "a very long detail string that should not blow past the viewport width"
        )
        rendered = step.render(sp_frame=0, width=60, label_width=24)
        self.assertLessEqual(cell_len(rendered.plain), 60)

    def test_knowledge_matches_are_width_limited(self) -> None:
        text = format_matches(
            [
                {
                    "path": "/very/long/path/to/a/file/that/keeps/going/and/going/example.py",
                    "score": 123.4,
                    "text": "this is a long excerpt that should get clipped for compact layouts",
                }
            ],
            width=50,
        )
        self.assertTrue(all(len(line) <= 50 for line in text.splitlines() if line))

    def test_weather_harness_output_wraps_for_narrow_widths(self) -> None:
        report = WeatherReport(
            location_label="Ashland, Kentucky, United States of America",
            lat="38.4780",
            lon="-82.6380",
            population="",
            queried_location="Ashland, Kentucky",
            desc="Clear",
            temp_f="62",
            feels_f="62",
            humidity="78",
            wind_mph="4",
            wind_dir="WSW",
            wind_deg="238",
            uv_index="0",
            visibility_mi="9",
            pressure_mb="1015",
            precip_in="0.0",
            cloud_cover="25",
            obs_time="02:48 AM",
            tz_name="America/New_York",
            forecast=[
                DayForecast(
                    date="2026-04-17",
                    desc="Sunny",
                    high_f="85",
                    low_f="58",
                    uv_index="7",
                    rain_chance=0,
                    snow_chance=0,
                    sunrise="06:42 AM",
                    sunset="08:02 PM",
                    moon_phase="Waning Crescent",
                    moon_illumination="25",
                    hourly=[
                        HourlySlot(
                            time_label="00:00",
                            temp_f="64",
                            feels_f="64",
                            desc="Clear",
                            wind_mph="5",
                            wind_dir="SW",
                            rain_chance=0,
                            snow_chance=0,
                            uv_index="0",
                            humidity="70",
                            cloud_cover="10",
                            is_current=True,
                        )
                    ],
                )
            ],
        )
        previous = MemorySnapshot(
            date="2026-04-16",
            location_key="ashland-kentucky",
            location_label="Ashland, Kentucky",
            queried_location="Ashland, Kentucky",
            desc="Clear",
            temp_f="70",
            feels_f="70",
            humidity="61",
            wind_mph="5",
            wind_dir="SW",
            uv_index="0",
            precip_in="0.0",
            cloud_cover="0",
            obs_time="02:48 AM",
        )
        comparison = WeatherComparison(
            found=True,
            previous_date="2026-04-16",
            summary="Temp down 8F.",
            bullets=["Temp down 8F", "Humidity up 17 pp", "Cloud cover up 25%"],
        )

        output = build_weather_harness_output(
            report,
            comparison,
            previous,
            "CURRENT: Clear skies.\nCHANGES: Cooler.\nHOURLY: Dry.\nOUTLOOK: Fair.",
            "",
            width=56,
        )
        self.assertTrue(all(len(line) <= 56 for line in output.splitlines() if line))


if __name__ == "__main__":
    unittest.main()
