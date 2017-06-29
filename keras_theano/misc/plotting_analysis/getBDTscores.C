#include <TTree.h>
#include <iostream>
#include <TFile.h>
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

using namespace std;

int main(int argc, char **argv)
{
    //takes 3 arguments,
    //full file path of ROOT file to read BDT data from
    //file name of text file containing gamma classifier values
    //file name of text file containing proton classifier values
    char *root_file_path = argv[1];
    char *gamma_file_name = argv[2];
    char *proton_file_name = argv[3];

    int label;
    float classifier_value;

    //open the file
    TFile *f = TFile::Open(root_file_path);
    if (f == 0) {
        printf("Error: cannot open file\n");
        return 0;
    }

    // Create tyhe tree reader and its data containers
    TTreeReader myReader("TestTree", f);

    TTreeReaderValue<int> classID = TTreeReaderValue<int>(myReader,"classID");
    TTreeReaderValue<float> BDT_0 = TTreeReaderValue<float>(myReader,"BDT_0");

    ofstream gammafile (gamma_file_name);
    ofstream protonfile(proton_file_name);

    while (myReader.Next()) 
    {
        label = *classID;
        classifier_value = *BDT_0;

        if(label==0)
        {
            gammafile << (classifier_value + 1)/2;
            gammafile << "\n";
        }
        else if(label==1)
        {
            protonfile << (classifier_value+1)/2;
            protonfile << "\n";
        }
        else
        {
        }
    }


    gammafile.close();
    protonfile.close();

    return 0;
}
