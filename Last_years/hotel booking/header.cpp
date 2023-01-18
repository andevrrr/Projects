#include <iostream>
using namespace std;

void checkAvail ( int a, char b, int c[], int d[], int e[], int f[], int room_one[], int room_two[], int brd[]){  // this subrutine takes values, also user inputs and calculates how many rooms left and books rooms

    int p_one = room_one[0]; // creating p_one and p_two to put the the values of room_one/two to count how many rooms left
    int p_two = room_two[0];



    if (a == 1){ // if user chose single room

        for (int i = 0; i < b; i++){
            for (int j = 0; j < 10; j++){
                if (c[j] > 0 && b <= p_one){
                    e[j] = c[j];
                    brd[i] = 0;
                    brd[i] = e[j];
                    c[j] = 0;
                    room_one[0] = room_one[0] - 1;
                    break;
                }else if( c[9] == 0){
                    break;
                }else {
                    continue;
                }
            }
        }

        if (b == 10 && c[9] == 0){
           
        } else if (b > p_one){
            cout << "\n\tSorry, only " << p_one << " room(s) left\n";
        } else if (c[9] == 0){
            cout << "\n\tYou booked the last room(s)" << endl;
        }

    } else if (a == 2){ // if user chose double room

        for (int i = 0; i < b; i++){
            for (int j = 0; j < 10; j++){
                if (d[j] > 0 && b <= p_two){
                    f[j] = d[j];
                    d[j] = 0;
                    brd[i] = 0;
                    brd[i] = f[j];
                    room_two[0] = room_two[0] - 1;
                    break;
                }else if( d[9] == 0){
                    break;
                }else {
                    continue;
                }
            }
        }
        if (b == 10 && d[9] == 0){
           
        } else if (b > p_two){
            cout << "\n\tSorry, only " << p_two << " room(s) left\n";
        } else if (d[9] == 0){
            cout << "\n\tYou booked the last room(s)" << endl;
        }
    }


}

double date (int day1,int month1, int year1, int day2, int month2, int year2) // This subrutine is about time, it collects user data on arrival and departure dates and finds the difference between them.
{   int i;
    double nights;
    month1-= 1;
    month2-= 1;
    struct tm a = {0,0,0,day1,month1,year1};
    struct tm b = {0,0,0,day2,month2,year2}; 
    time_t x = mktime(&a);
    time_t y = mktime(&b);
    double days;
    if ( x != (std::time_t)(-1) && y != (time_t)(-1) )
    {
        double nights = difftime(y, x) / (60 * 60 * 24);
        cout << "\n  Number of Nights = " << nights << endl;
        days = nights;
    }

    return days;

}