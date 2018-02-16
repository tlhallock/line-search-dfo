//
// Created by thallock on 12/25/17.
//

#ifndef C_CONFIG_H
#define C_CONFIG_H

#include <string>


const std::string NOT_FOUND = "not found";

class ConfigInternals;

class Config {
    ConfigInternals *internals;

    void load();
public:
    Config();
    ~Config();

    std::string get_property(const std::string& key) const;
};

const Config& get_config_instance();

#endif //C_CONFIG_H
