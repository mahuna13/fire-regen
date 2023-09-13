import geopandas as gpd


def geo_plot(gdf: gpd.GeoDataFrame, ax, column, vmin=None, vmax=None,
             cmap='inferno_r', markersize=5):
    if vmin is None:
        vmin = gdf[column].quantile(0.05)
    if vmax is None:
        vmax = gdf[column].quantile(0.95)
    gdf.plot(column=column, ax=ax, legend=True, vmin=vmin,
             vmax=vmax, markersize=markersize, cmap=cmap)
