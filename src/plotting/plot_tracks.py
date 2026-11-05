import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0  # km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def get_probability_of_distance(init_group, LON, LAT, radius_km):
    n_members = init_group["ensemble"].nunique()
    probability = np.zeros(LON.shape)
    for ensemble, ensemble_df in init_group.groupby("ensemble"):
        # print(ensemble)

        # Get trajectory points
        lat = ensemble_df["latitude"].to_numpy()
        lon = ensemble_df["longitude"].to_numpy()

        # (n_segments, 2)
        A = np.column_stack((lon[:-1], lat[:-1]))
        B = np.column_stack((lon[1:], lat[1:]))

        mask = np.zeros(LON.shape, dtype=bool)

        for (x1, y1), (x2, y2) in zip(A, B):
            vx = x2 - x1
            vy = y2 - y1

            wx = LON - x1
            wy = LAT - y1

            vv = vx * vx + vy * vy

            if vv < 1e-12:
                continue

            t = (wx * vx + wy * vy) / vv
            t = np.clip(t, 0, 1)

            cx = x1 + t * vx
            cy = y1 + t * vy

            dist = haversine(LAT, LON, cy, cx)

            mask |= dist <= radius_km

        probability += mask.astype(float)

    probability /= n_members
    return probability


def get_continent_info(lat, lon):
    """
    Determine the continent/region containing the center of the track.

    Returns:
        continent_name, [lon_min, lon_max, lat_min, lat_max]
    """

    center_lat = np.mean(lat)
    center_lon = np.mean(lon)

    # Australia
    if -45 <= center_lat <= -10 and 110 <= center_lon <= 155:
        return "Australia", [110, 155, -45, -5]

    # North America
    elif 5 <= center_lat <= 85 and -170 <= center_lon <= -50:
        return "North America", [-170, -50, 5, 85]

    # South America
    elif -60 <= center_lat <= 15 and -85 <= center_lon <= -30:
        return "South America", [-85, -30, -60, 15]

    # Europe
    elif 35 <= center_lat <= 72 and -25 <= center_lon <= 45:
        return "Europe", [-25, 45, 35, 72]

    # Africa
    elif -40 <= center_lat <= 40 and -20 <= center_lon <= 55:
        return "Africa", [-20, 55, -40, 40]

    # Asia
    elif 0 <= center_lat <= 80 and 40 <= center_lon <= 180:
        return "Asia", [40, 180, 0, 80]

    # Default
    else:
        return "World", [-180, 180, -60, 90]


def plot_map_real(
    df_predict, lat, lon, lon_max, lon_min, lat_max, lat_min, out_dir, radius_km
):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "font.size": 13,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
        }
    )

    df_predict = df_predict.reset_index()

    df_predict["lead_time"] = pd.to_timedelta(df_predict["lead_time"])

    df_predict = df_predict.sort_values(["init_time", "lead_time"])
    df_predict = df_predict.set_index(["init_time", "lead_time"])
    df_predict = df_predict.sort_values(["init_time", "ensemble", "lead_time"])

    borders = 5

    lon_grid = np.linspace(lon_min - borders, lon_max + borders, 400)
    lat_grid = np.linspace(lat_min - borders, lat_max + borders, 400)

    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    for init_time, init_group in df_predict.groupby("init_time"):
        # print(init_time)
        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------

        fig = plt.figure(figsize=(7, 4.8))

        # ax = plt.axes(projection=ccrs.PlateCarree())

        ax = fig.add_axes([0.08, 0.16, 0.68, 0.74], projection=ccrs.PlateCarree())

        ax.set_extent(
            [lon_min - borders, lon_max + borders, lat_min - borders, lat_max + borders]
        )

        ax.add_feature(cfeature.LAND, facecolor="0.92")
        ax.add_feature(cfeature.OCEAN, facecolor="#dceeff")
        ax.coastlines("50m")

        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        fig.canvas.draw()

        # ------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------

        ax.plot(
            lon,
            lat,
            color="red",
            linewidth=2.5,
            label="Ground Truth",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

        ax.scatter(
            lon[0],
            lat[0],
            color="lime",
            edgecolor="black",
            s=120,
            label="Start",
            transform=ccrs.PlateCarree(),
            zorder=20,
        )

        ax.scatter(
            lon[-1],
            lat[-1],
            marker="*",
            color="lime",
            edgecolor="black",
            s=200,
            label="End",
            transform=ccrs.PlateCarree(),
            zorder=20,
        )

        # ------------------------------------------------------------
        # Heatmap
        # ------------------------------------------------------------

        # Mask all zero-probability pixels
        probability = get_probability_of_distance(init_group, LON, LAT, radius_km)

        masked_probability = np.ma.masked_where(probability == 0, probability)

        levels = np.linspace(0, 1, 21)

        cf = ax.contourf(
            LON,
            LAT,
            masked_probability,
            levels=levels,
            cmap="turbo",
            transform=ccrs.PlateCarree(),
        )

        # ------------------------------------------------------------
        # Mean track
        # ------------------------------------------------------------
        mean_track = (
            init_group.groupby("lead_time")[["latitude", "longitude"]]
            .mean()
            .sort_index()
        )

        ax.plot(
            mean_track["longitude"],
            mean_track["latitude"],
            color="black",
            linewidth=2.5,
            label="Mean prediction",
            transform=ccrs.PlateCarree(),
        )
        # print(mean_track)

        # ------------------------------------------------------------
        # Colorbar
        # ------------------------------------------------------------

        cbar = plt.colorbar(cf, ax=ax, shrink=0.85)
        cbar.set_label("Probability", fontsize=13)
        cbar.ax.tick_params(labelsize=12)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=4,
            fontsize=13,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        # =========================================================
        # Inset Map
        # =========================================================
        continent, inset_extent = get_continent_info(lat, lon)

        # Position in figure coordinates:
        # [left, bottom, width, height]
        # Australia inset
        ax_inset = fig.add_axes([0.45, 0.23, 0.15, 0.18], projection=ccrs.PlateCarree())

        # Zoomed-out view of the continent
        ax_inset.set_extent(inset_extent, crs=ccrs.PlateCarree())

        # --- Typical map colors ---
        ax_inset.stock_img()

        # Coastline on top
        ax_inset.coastlines(linewidth=0.8, edgecolor="black", zorder=5)

        # =========================================================
        # Highlight the region shown in the main plot
        # =========================================================

        main_extent = [
            lon_min - borders,  # west
            lon_max + borders,  # east
            lat_min - borders,  # south
            lat_max + borders,  # north
        ]

        rect = mpatches.Rectangle(
            (main_extent[0], main_extent[2]),
            main_extent[1] - main_extent[0],
            main_extent[3] - main_extent[2],
            fill=False,
            edgecolor="red",
            linewidth=2,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

        ax_inset.add_patch(rect)

        # =========================================================
        # "Australia" label underneath the inset
        # =========================================================

        ax_inset.text(
            0.5,
            -0.03,
            continent,
            transform=ax_inset.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            color="black",
            clip_on=False,
        )

        # =========================================================
        # Remove ticks from inset
        # =========================================================

        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        for spine in ax_inset.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"cyclone_track_{init_time}_{radius_km}km"

        fig.savefig(
            out_dir / f"{filename}.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        fig.savefig(
            out_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--ensemble_predictions", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--radius", type=float, default=150)
    args = parser.parse_args()
    out_dir = args.output_dir
    radius_km = args.radius

    df = pd.read_csv(args.ground_truth)
    df_predict = pd.read_csv(args.ensemble_predictions)

    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    lat_min = min(lat)
    lat_max = max(lat)
    lon_min = min(lon)
    lon_max = max(lon)

    plot_map_real(
        df_predict, lat, lon, lon_max, lon_min, lat_max, lat_min, out_dir, radius_km
    )


if __name__ == "__main__":
    main()
