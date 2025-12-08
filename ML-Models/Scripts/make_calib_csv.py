import pandas as pd

# Load your clicks
df_px = pd.read_csv("clicked_pixels.csv")  # columns: x_px,y_px

# EDIT the list below to give the (x_m, y_m) for each click IN THE SAME ORDER
# Example for [far-left corner, far-right corner, net×left sideline, net×right sideline]:
meter_coords = [
    (3.0, 0.0),   # click #1
    (3.0, 5.0),  # click #2
    (3.0, 10.0),  # click #3
    (10.0, 0.0), # click #4
    (10.0, 10.0), # click #5
    (17.0, 0.0), # click #6
    (17.0, 5.0), # click #7
    (17.0, 10.0), # click #8
    # add more tuples if you clicked more points
]

assert len(meter_coords) == len(df_px), "Number of meter coords must match number of clicks"
df_m = pd.DataFrame(meter_coords, columns=["x_m","y_m"])
df = pd.concat([df_px, df_m], axis=1)
df.to_csv("calib_points.csv", index=False)
print("Wrote calib_points.csv")
print(df)
