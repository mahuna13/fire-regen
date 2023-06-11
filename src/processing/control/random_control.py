
from src.data import fire_perimeters
import geopandas as gpd

# Algo 1 - for finding control shots.


def match_with_random_shots_per_fire(
    fire: fire_perimeters.Fire,
    gedi: gpd.GeoDataFrame,
    buffer_size: int,
    num_samples: int
) -> gpd.GeoDataFrame:
    buffer = fire.get_buffer(buffer_size, 100)

    # Find unburned shots within the buffer.
    within_buffer = gedi.sjoin(
        buffer, how="inner", predicate="within")

    within_fire = gedi.sjoin(
        fire.fire, how="inner", predicate="within")

    # Pick 200 random shots and assign as control. Do both median and mean.
    random_controls = within_buffer.sample(
        min(num_samples, within_buffer.shape[0]))
    within_fire['agbd_control_mean'] = random_controls.agbd.mean()
    within_fire['agbd_control_median'] = random_controls.agbd.median()

    return within_fire


# Algo 2 - for finding control shots.
def match_with_random_shots_per_pixel(
    fire: fire_perimeters.Fire,
    gedi: gpd.GeoDataFrame,
    buffer_size: int,
    num_samples: int
) -> gpd.GeoDataFrame:
    buffer = fire.get_buffer(buffer_size, 100)

    # Find unburned shots within the buffer.
    within_buffer = gedi.sjoin(
        buffer, how="inner", predicate="within")

    within_fire = gedi.sjoin(
        fire.fire, how="inner", predicate="within")

    def get_control_for_pixel(row):
        # Pick 200 random shots and assign as control. Do both median and mean.
        random_controls = within_buffer.sample(
            min(num_samples, within_buffer.shape[0]))
        row['agbd_control_mean'] = random_controls.agbd.mean()
        row['agbd_control_median'] = random_controls.agbd.median()
        return row

    within_fire = within_fire.apply(get_control_for_pixel, axis=1)

    return within_fire
