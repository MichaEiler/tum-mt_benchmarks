#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

bool ReadFile(const string& name, vector<char>& data) {
    ifstream stream(name, ios::in);
    if (stream) {
        stream.seekg(0, ios::end);
        data.resize(stream.tellg());
        stream.seekg(0, ios::beg);
        stream.read(&data[0], data.size());
        stream.close();
        return true;
    }
    cerr << "Could not open input file..." << endl;
    return false;
}

bool WriteFile(const string& name, char* data, size_t length) {
    fstream stream(name, ios::binary | ios::out);

    if (!stream) {
        cerr << "Could not open output file..." << endl;
        return false;
    }

    stream.write(&data[0], length);
    stream.close();

    return true;
}

bool SplitFile(vector<char>& data, const string& splitStr, const string& outputFileName) {
    int outputFileId = 0;
    size_t start = 0;

    for (size_t i = 1; i < data.size() - splitStr.length(); ++i) {
        for (size_t j = 0; j < splitStr.length(); ++j) {
            if ( data[i + j] != splitStr[j]) {
                break;
            } else if ( j == (splitStr.length() - 1)){
                if (!WriteFile(outputFileName + "." + to_string(outputFileId), &data[start], (i - start))) {
                    return false;
                }

                outputFileId++;
                start = i;
            }
        }
    }

    if (!WriteFile(outputFileName + "." + to_string(outputFileId), &data[start], (data.size() - start + 1))) {
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Not enough arguments. Call with ./elfsplit inputFile <str> outputFilePrefix" << endl;
        return -1;
    }

    string inputFile(argv[1]), splitStr(argv[2]), outputFilePrefix(argv[3]);

    vector<char> inputData;
    if (!ReadFile(inputFile, inputData)) {
        return -1;
    }

    if (!SplitFile(inputData, splitStr, outputFilePrefix)) {
        return -1;
    }

    return 0;
}
