from configparser import ConfigParser

# Get the configparser object
config_object = ConfigParser()

# Define the config object
config_object["TESTING"] = {
    "using_real_time_models": True,
    "plot_real_time_graph": True,
}

# Write to config file
with open('config.ini', 'w') as conf:
    config_object.write(conf)


