//
// Created by thallock on 12/25/17.
//

#include "Config.h"

#include <boost/algorithm/string.hpp>
#include <vector>

#include <cstdio>
#define RAPIDJSON_NAMESPACE rj
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>


static const std::string filename = "/work/research/line-search-dfo/c/config.json";

class ConfigInternals {
public:
    rj::Document document;
};


Config::Config() {
    internals = new ConfigInternals();
    load();
}

Config::~Config() {
    delete internals;
}

void Config::load() {
    FILE* pFile = fopen(filename.c_str(), "rb");
    char buffer[65536];
    rj::FileReadStream is(pFile, buffer, sizeof(buffer));
    internals->document.ParseStream<0, rj::UTF8<>, rj::FileReadStream>(is);
    fclose(pFile);
}

std::string Config::get_property(const std::string& key) const {
    std::vector<std::string> results;
    boost::split(results, key, [](char c){return c == '.';});

    rj::GenericValue<rj::UTF8<> > *current = &internals->document;
    for (auto it = results.begin(); it != results.end(); ++it) {
        current = &(*current)[it->c_str()];
    }

    return std::string(current->GetString());
}

static Config config;
const Config& get_config_instance() {
    return config;
}