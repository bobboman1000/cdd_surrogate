from functools import reduce
import re

import pandas as pd

from fccd.deformation import estimate_stress_plastic_deformation_point, calculate_rho_total, DISLOCATION_MOVEMENT_METHOD

# elastic stiffnesses for Aluminium from Rösler,Harders,Bäker_2016_Book_Mechanisches Verhalten der Werkstoffe
# Al (A=1,23,E_iso=70): C_1111, C_1122, C_2323 = 108, 61, 29
# Au (A=1,89,E_iso=78): C_1111, C_1122, C_2323 = 186, 157, 42
# Cu (A=3,22,E_iso=121): C_1111, C_1122, C_2323 = 168, 121, 75


# Just for mapping the materials. If you need to add a material add it here with its parameters.
MATERIAL_STIFFNESS = {
    # [C1111, C1122, C2323]: Material
    (108, 61, 29): "Al",
    (186, 157, 42): "Au",
    (168, 121, 75): "Cu"
}

E = {
    # [C1111, C1122, C2323]: Material
    "Al": 72.738,
    "Au": 78,
    "Cu": 117
}

NU = {
    # [C1111, C1122, C2323]: Material
    "Al": 0.347,
    "Au": 0.44,
    "Cu": 0.35
}


# This is the list of keys which are read from the logfile. If it's listed here it will appear in params dict.
# You will still need to map it in the map_params function.

parameters_to_collect_in_log = [
    "Mesh",                                                         # 1. Mesh
    "C_1111", "C_1122", "C_2323", "burger", "E", "nu",               # 2. Material
    "rho0", "rho_S0", "rho_lomer0",                                 # 3 + 4 Versetzungsdichte
    "euler_angles",                                                 # 5. Kristallorientierung
    "c_FR",                                                         # 6. Quelllängen-Faktor
    "c_FR_multi",                                                   # 7. Ausbauch Faktor
    "lomer_const",                                                  # 8. Lomer constant
    "coll_const",                                                   # 8. Collision constant
    "tau_stageIII",
    "prefactor_V_act"
]


def euler_string(angles: str):
    if type(angles) == list:
        angles = reduce(lambda a, b: a + b, angles)

    concat_angles = angles.replace(" ", "")
    angle_description = {
        "0.000000.000000.00000": "e100",
        "0.000000.00000-45.00000": "e110",
        "-45.0000035.264400.00000": "e111",
        "26.5651053.300800.00000": "e123"
    }
    return angle_description[concat_angles]


def map_params(parameters: dict) -> dict:
    """
    Specify how parameters should be mapped. Everything mapped here will appear in the final df.
    :param parameters: parameter map read from the logfile
    :return: ´version of the map that ONLY contains what should end up in the df with its column name
    """
    mapped_parameters = {}

    c = float(parameters["C_1111"]), float(parameters["C_1122"]), float(parameters["C_2323"])
    material = MATERIAL_STIFFNESS[c]

    mapped_parameters["mesh"] = parameters["Mesh"].split("_")[1][3:] # Does both

    mapped_parameters["material"] = material

    burger_vektor_element = parameters["burger"][0]
    mapped_parameters["burger"] = burger_vektor_element

    mapped_parameters["nu"] = parameters["nu"]
    mapped_parameters["E"] = parameters["E"]

    rho_tot0 = calculate_rho_total(parameters["rho0"], parameters["rho_S0"], parameters["rho_lomer0"])
    mapped_parameters["rho_tot0"] = rho_tot0

    mapped_parameters["rho0"] = parameters["rho0"]
    mapped_parameters["rho_S0"] = parameters["rho_S0"]
    mapped_parameters["rho_lomer0"] = parameters["rho_lomer0"]

    euler_angles = euler_string(parameters["euler_angles"])
    mapped_parameters["euler_angles"] = euler_angles

    mapped_parameters["c_FR"] = float(parameters["c_FR"])
    mapped_parameters["c_FR_multi"] = float(parameters["c_FR_multi"])
    mapped_parameters["lomer_const"] = float(parameters["lomer_const"])
    mapped_parameters["coll_const"] = float(parameters["coll_const"])
    mapped_parameters["tau_stageIII"] = float(parameters["tau_stageIII"])
    mapped_parameters["prefactor_V_act"] = float(parameters["prefactor_V_act"])

    plastic_deformation_point = estimate_stress_plastic_deformation_point(
        method=DISLOCATION_MOVEMENT_METHOD,
        E=float(mapped_parameters["E"]),
        nu=float(mapped_parameters["nu"]),
        rho_lomer0=float(mapped_parameters["rho_lomer0"]),
        rho_0=float(mapped_parameters["rho0"]),
        rho_S0=float(mapped_parameters["rho_S0"]),
        burger_vektor=float(mapped_parameters["burger"]),
        crystal_orientation=mapped_parameters["euler_angles"],
        c_fr=float(mapped_parameters["c_FR"]),
        rho_tot=float(mapped_parameters["rho_tot0"]),
        c_multi=float(mapped_parameters["c_FR_multi"])
    )
    mapped_parameters["psd"] = plastic_deformation_point

    return mapped_parameters


def parse_logfile(path_to_experiment):
    """
    Parse logfile. Collect all vars that are contained in parameter_keys_log.
    :param path_to_experiment: Path containing all experiments (expects logfile under PATH/data/log)
    :return: dict containing all params
    """
    parameters = dict()
    with open(path_to_experiment + "/log", "r") as input_file_1:
        for line in input_file_1:
            if any(["reading ... " + param_key in line for param_key in parameters_to_collect_in_log]):
                _, declaration = line.split("reading ... ")

                declaration = declaration.replace("= ", "=")
                declaration = declaration.replace(" =", "=")
                declaration = declaration.strip()

                key, value = declaration.split("=")

                tokens = value.split(" ")
                if len(tokens) > 1:
                    value = tokens

                if key in parameters_to_collect_in_log:
                    parameters[key] = value

                if len(parameters.keys()) >= len(parameters_to_collect_in_log):
                    break  # Stop when all are found

            if re.match("^ *[Enu].{1,2} *= *[0-9.]+ *$", line) is not None:
                declaration = line.replace(" ", "")
                key, value = declaration.split("=")
                if key in parameters_to_collect_in_log:
                    parameters[key] = value
                if len(parameters.keys()) >= len(parameters_to_collect_in_log):
                    break  # Stop when all are found

    return parameters


def extract_params(path):
    """
    Parse variables from logfile, map params using map_params and save it as a df.
    :param path: Path containing all experiments (expects logfile under PATH/data/log)
    :return: Returns df, containing the parameters
    """

    logfile_parameters = parse_logfile(path)
    mapped_parameters = map_params(logfile_parameters)
    df = pd.DataFrame(mapped_parameters, index=[1,])
    df.to_csv(path + "/params.csv")
    return df


if __name__ == "__main__":
    extract_params("")

