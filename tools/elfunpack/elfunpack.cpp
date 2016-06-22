#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <bfd.h>

using namespace std;

class ElfContainer {
private:
    bfd *_container = nullptr;
public:
    explicit ElfContainer() { }
    virtual ~ElfContainer() { Close(); }

    bool Open(const string& inputFile) {
        _container = bfd_openr(inputFile.c_str(), nullptr);
        if (_container == nullptr) {
            cerr << "Could not open elf file." << endl;
            return false;
        }

        if (!bfd_check_format(_container, bfd_object)) {
            if (bfd_get_error() != bfd_error_file_ambiguously_recognized) {
                cerr << "File format error." << endl;
                bfd_close(_container);
                _container = nullptr;
                return false;
            }  
        }

        return true;
    }

    bool ReadSection(const string& sectionName, vector<char>& data) {
        asection *selectedSection = bfd_get_section_by_name(_container, sectionName.c_str());
        if (selectedSection == nullptr) {
            cerr << "Section " << sectionName << " not found." << endl;
            return false;
        }

        size_t sectionSize = bfd_section_size(_container, selectedSection);
        data.resize(sectionSize);

        if (!bfd_get_section_contents(_container, selectedSection, &data[0], 0, sectionSize)) {
            cerr << "Failed to retrieve section contents." << endl;
            return false;
        }

        return true;
    }

    void Close() {
        if (_container != nullptr) {
            bfd_close(_container);
            _container = nullptr;
        }
    }

    ElfContainer(const ElfContainer&) = delete;
    ElfContainer& operator=(const ElfContainer&) = delete;
};

class OutputStream {
public:
    static bool Write(const string& name, char* data, size_t length) {
        fstream stream(name, ios::binary | ios::out);

        if (!stream) {
            cerr << "Could not open output file..." << endl;
            return false;
        }

        stream.write(&data[0], length);
        stream.close();

        return true;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Not enough arguments. Call with ./elfunpack inputFile sectionName outputFile" << endl;
        return -1;
    }

    string inputFile(argv[1]), sectionName(argv[2]), outputFile(argv[3]);

    ElfContainer container;
    vector<char> data;

    if (!container.Open(inputFile)) {
        return -1;
    }
    
    if (!container.ReadSection(sectionName, data)) {
        return -1;
    }

    if (!OutputStream::Write(outputFile, &data[0], data.size())) {
        return -1;
    }

    return 0;
}
