import pandas as pd


def get_PS129_CTD_data():
    # get location of CTD files from the LADCP data files
    LADCP_DIRECTORY = f"/media/sf_VM_Folder/figures/PS129_Plots/ladcp_profiles/"
    ladcp_paths = sorted(
        helper.IO.get_filepaths_from_directory(LADCP_DIRECTORY, inclusive=(".mat",))
    )
    assert len(ladcp_paths) != 0

    # all CTD cast can be identified by a number
    ladcp_cast_numbers = [path.split("/")[-1][7:-4] for path in ladcp_paths]

    # create list of all CTD cast locations
    ctd_locations = []
    ctd_timestamps = []
    for path in ladcp_paths:
        data_struct = sio.loadmat(path)
        data = data_struct["dr"][0][0]
        lat = np.round(np.squeeze(data["lat"]), 3)
        lon = np.round(np.squeeze(data["lon"]), 3)
        ctd_locations.append(Location(lat=lat, lon=lon))
        time_stamp = datetime.datetime(
            *map(int, np.squeeze(data["date"]))
        )  # convert list of values to datetime object
        ctd_timestamps.append(time_stamp)

    name_to_number_dict = load_stations("/media/sf_VM_Folder/figures/PS129_Plots/conversion.txt")
    number_to_name_dict = {int(v): k for k, v in name_to_number_dict.items()}

    # create as many CTDCast objects as casts itself
    list_of_PS129_casts = [CTDCast() for _ in ladcp_cast_numbers]

    # for every cast object set a location and a name
    for i, cast in enumerate(list_of_PS129_casts):
        cast_number = ladcp_cast_numbers[i]
        cast_name = number_to_name_dict[int(cast_number)]
        cast.name = cast_name
        cast.location = ctd_locations[i]
        cast.date = ctd_timestamps[i]

    column_names = [
        "Event",
        "Latitude",
        "Longitude",
        "Press [dbar]",
        "Sal",
        "Temp [°C]",
        "Absolute Salinity",
        "Conservative Temperature",
        "Date/Time",
        "Depth water [m]",
        "Expedition",
    ]
    data_dict = {name: [] for name in column_names}

    for cast in list_of_PS129_casts:
        # load actual data to that Cast name
        try:
            path = f"/media/sf_VM_Folder/figures/PS129_Plots/ctd_profiles/dps129_{cast.name}.cnv"

            # SP = Practical Salinity [PSU]
            # Temperature [ITS-90, °C]

            # add the cast data to the data dictionary
            pressure, in_situ_temperature, practical_salinity = np.genfromtxt(
                path, skip_header=322, usecols=(0, 1, 5), unpack=True
            )
            data_dict["Event"].extend([f"PS129_{cast.name}"] * len(pressure))
            data_dict["Latitude"].extend([cast.location.lat] * len(pressure))
            data_dict["Longitude"].extend([cast.location.lon] * len(pressure))
            data_dict["Press [dbar]"].extend(pressure)
            data_dict["Temp [°C]"].extend(in_situ_temperature)
            data_dict["Sal"].extend(practical_salinity)
            data_dict["Date/Time"].extend([cast.date] * len(pressure))
            data_dict["Expedition"].extend(["PS129"] * len(pressure))

            # add new attributes
            SA = gsw.SA_from_SP(
                SP=practical_salinity, p=pressure, lon=cast.location.lon, lat=cast.location.lat
            )
            data_dict["Absolute Salinity"].extend(SA)
            data_dict["Conservative Temperature"].extend(
                gsw.CT_from_t(SA=SA, t=in_situ_temperature, p=pressure)
            )
            data_dict["Depth water [m]"].extend(
                np.abs(gsw.z_from_p(p=pressure, lat=cast.location.lat))
            )

        except ValueError as e:
            print("ValueError at ", cast.name, cast.location)
            print(e)
            continue
        # print(cast.name)
    # print(data_dict)
    for k, v in data_dict.items():
        print(k, len(v))
    return pd.DataFrame(data=data_dict)


def load_Joinville_transect_CTDs():
    # try the file with neutral densities (matlab output)
    path = "data/CTD/joinville_transect_ctds_incl_neutral_density.csv"
    for _ in range(6):  # try to find the right folder only five times to avoid an endless loop
        try:
            CTDs = pd.read_csv(path)

        # try from parent directory
        except FileNotFoundError as error:
            path = "../" + path
            continue

        else:
            print(f"loading of {path} was successful")
            print("renaming of matlab style columns")
            new_columns = [
                "index",
                "Event",
                "Latitude",
                "Longitude",
                "Press [dbar]",
                "Sal",
                "Temp [°C]",
                "Absolute Salinity",
                "Conservative Temperature",
                "Date/Time",
                "Depth water [m]",
                "Expedition",
                "Neutral density [kg m^-3]",
            ]
            for i, j in zip(CTDs.columns, new_columns):
                print(i, "\t", j)
            CTDs.columns = new_columns
            CTDs.set_index("index", inplace=True)
            return CTDs

    print("CTD data including neutral densities could not be found")

    # try the file without neutral density
    path = "data/CTD/joinville_transect_ctds.csv"
    for _ in range(6):  # try to find the right folder only five times to avoid an endless loop
        try:
            CTDs = pd.read_csv(path)
        except FileNotFoundError as error:
            path = "../" + path
            continue

        else:
            print(f"loading of {path} was successful")
            if "neutral_density" not in CTDs.columns:
                print(
                    "!!! Warning: !!!\n"
                    "neutral_density is missing and has to be computed with the corresponding matlab toolbox\n"
                    "This variable may be required for some code to run successfully"
                    ""
                )

            return CTDs

    print("joinville_transect_ctds could not be found")
    print(f"last checked at {path}")
    print(
        "Did you preprocess the raw CTD profiles first by running save_Joinville_transect_CTDs_to_csv() ?"
    )
    raise FileNotFoundError
