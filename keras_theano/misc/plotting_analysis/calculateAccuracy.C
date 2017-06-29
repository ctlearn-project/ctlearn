#include <TTree.h>
#include <iostream>
#include <TFile.h>
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

using namespace std;

int main(int argc, char **argv)
{
    int label;
    float classifier_value;

    int signal = 0;
    int background = 0;
    int signal_correct = 0;
    int signal_incorrect = 0;
    int background_correct = 0;
    int background_incorrect = 0;

    //open the file
    TFile *f = TFile::Open("/data/nieto/TMVA/BDT-V2-ID0-IM3-d20170517-0.5/complete_BDTroot/BDT_0.root");
    if (f == 0) {
        printf("Error: cannot open file\n");
        return 0;
    }

    // Create tyhe tree reader and its data containers
    TTreeReader myReader("TestTree", f);

    TTreeReaderValue<int> classID = TTreeReaderValue<int>(myReader,"classID");
    TTreeReaderValue<float> BDT_0 = TTreeReaderValue<float>(myReader,"BDT_0");

    while (myReader.Next()) {
        label = *classID;
        classifier_value = *BDT_0;

        if((classifier_value < 0.0) && (label==1))
        {
            background++;
            background_correct++;
        }
        else if((classifier_value >= 0.0) && (label==0))
        {
            signal++;
            signal_correct++;
        }
        else if((classifier_value >= 0.0) && (label==1))
        {
            background++;
            background_incorrect++;
        }
        else if((classifier_value < 0.0) && (label==0))
        {
            signal++;
            signal_incorrect++;
        }
        else
        {
        }
    }

    int total_correct = signal_correct+background_correct;
    int total_incorrect = signal_incorrect+background_incorrect;
    int total = total_correct + total_incorrect;
    float accuracy = float(total_correct)/float(total);

    printf("Total number of events: %d \n\n",total);

    printf ("Number of signal: %d \n", signal);
    printf ("Number of signal correctly classified: %d \n", signal_correct);
    printf ("Number of signal incorrectly classified: %d \n\n", signal_incorrect);

    printf ("Number of background: %d \n", background);
    printf ("Number of background correctly classified: %d \n", background_correct);
    printf ("Number of background incorrectly classified: %d \n\n", background_incorrect);

    printf ("Total correctly classified: %d \n", total_correct);
    printf ("Total incorrectly classified: %d \n\n", total_incorrect);
    
    printf ("Accuracy: %f \n",accuracy);

    return 0;
}

