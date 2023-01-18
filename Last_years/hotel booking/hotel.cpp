#include <iostream>
#include "header.h"
#include "header.cpp"
using namespace std;



struct information {

    string name;
    string surname;
    int age;
    int passport_number;
    int phone_number;

    int day1, month1, year1;
    int day2, month2, year2;
    int number_of_rooms;
    int ref_number;
    int room_size; // there are only two sizes: single and double
    int booked_room[16];
    string user_input_two;

};


int main(){

    
    information person;
    information person_book[20]; // size of this array should be that to be enough for max people that can book a rooms, in my case it is 20 as i have 20 rooms, means 20 people can book each room.
    information person_details[20]; // same as above 

    for (int i = 0; i < 20; i++){ // here i want just to be sure that each value in the arrays is zero
        for (int j = 0; j < 20; j++){
            person_book[i].booked_room[j] = 0;
        }
    }

    string user_input; // this one is just for 231 line to ask a user for input
    int room_num_single[10] = {0,0,0,0,0,0,0,0,0,0};  // this.. 
    int room_num_double[10] = {0,0,0,0,0,0,0,0,0,0};  // ..two arrays are needed just to store booked rooms, in this program i do not use it somehow for output but it can be used to show which and how many rooms are booked to the vendor.
    int single_rooms[10] = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110}; 

    /* 

    if there are more than 10 rooms in the hotel, i can create the following function, i did not implement it in my program,
    as i do not want to work with big numbers, and it easier to test when the range is not long.

    int room_num_single[100];

    for (int i = 0; i < 100; i++){
        int b = 101;
        room_num_single[i] = b + i;
    }
    
    */


    int double_rooms[10] = {111, 112, 113, 114, 115, 116, 117, 118, 119, 120};
    int room_single_left[1] = {10}; // to count how many rooms left
    int room_double_left[1] = {10};
    int booked_rooms_dates[10] = {0,0,0,0,0,0,0,0,0,0}; // stores booked rooms after user input of how many rooms he\she wants to book and when it is a turn of another person, this array will completely change for another parametrs.


    cout << "\n";
    cout << "\n\t\t\t\t\t\tHOTEL BOOKING SERVICE";
    cout << "\n" << "\n" << "\n";

    cout << "\nRoom rates\n";
    cout << "\n· Single room: 100 euros per night\n\n";
    cout << "\n· Double room: 150 euros per night.\n\n\n";

    cout <<"\n\t*****DISCOUNTS*****\n\n";
    cout << "\n * If you book 2 single rooms, you will get 10% discount on final price\n\n";
    cout << "\n * If you book 2 double rooms, you will get 5% discount on final price\n\n";

    int num = 0; // i need it for the array in do-shile function

    start:

    bool reg = true;
    

    do { // here each circle of the function it is a new input, new booked room, that is why to somehow store what rooms first user booked, second user, etc, i needed to add an array inside of array, first array counts users and inside array is about what and how many rooms the user booked.
   
        num = num + 1;

        cout << "\n\tEnter the arrival date\n\n";

        cout << "Day: ";
        cin >> person_details[num].day1;
        cout << "Month: ";
        cin >> person_details[num].month1;
        cout << "Year: ";
        cin >> person_details[num].year1;


        cout << "\n\tEnter the departure day\n\n";

        cout << "Day: ";
        cin >> person_details[num].day2;
        cout << "Month: ";
        cin >> person_details[num].month2;
        cout << "Year: ";
        cin >> person_details[num].year2;


        cout << "\n\tEnter the size of the room\n\n";
        cout << " '1' is a single room, '2' is a double room\n";
        cin >> person.room_size;

        person_book[num].room_size = person.room_size;

        cout << "\n\tHow many rooms do you need?\n\n";
        cin >> person.number_of_rooms;

        person_book[num].number_of_rooms = person.number_of_rooms;

        checkAvail (person.room_size, person.number_of_rooms, single_rooms, double_rooms, room_num_single, room_num_double, room_single_left, room_double_left,booked_rooms_dates);

        for (int i = 0; i < person.number_of_rooms; i++){
            person_book[num].booked_room[i] = booked_rooms_dates[i];
        }

        if (reg == true) {
            cout << "\n\tEnter your name: \n";
            cin >> person_details[num].name;
            cout << "\n\tEnter your surname: \n";
            cin >> person_details[num].surname;
            cout << "\n\tEnter your passport number: \n";
            cin >> person_details[num].passport_number;
            cout << "\n\tEnter your age: \n";
            cin >> person_details[num].age;
            cout << "\n\tEnter your phone number: \n";
            cin >> person_details[num].phone_number;
            person_details[num].ref_number = (rand()%89999) + 10000;
        }

                
        cout << "\n\tDo you need more rooms? Enter Yes or No\n\n";
        cin >> user_input;


        if (user_input == "Yes" || user_input == "yes") {
            cout << "Do you want to register another room in your name? ";
            cin >> person_details[num].user_input_two;
            if (person_details[num].user_input_two == "Yes" || person_details[num].user_input_two == "yes"){
                reg = false;
            } else if (person_details[num].user_input_two == "No" || person_details[num].user_input_two == "no") {
                reg = true;
            }
        } 

        

    } while( user_input == "Yes" || user_input == "yes");

    
    cout << "\n\tYou succesfully booked the room(s)\n\n";

    cout << "\n\t****Booking information****\n"; // first output is not inside of next function (200 line) because first output should print user informatin (name, surname etc) and if i put it inside of the function and in line 236 i will choose yes, in array num = 1, means that num[1] = yes, that will not show person's details.

    cout << "\n\t Name: ";
    cout << person_details[1].name;
    cout << "\n\t Surname: ";
    cout <<person_details[1].surname;
    cout << "\n\t Passport number: ";
    cout << person_details[1].passport_number;
    cout << "\n\t Age: ";
    cout << person_details[1].age;
    cout << "\n\t Phone number: ";
    cout << person_details[1].phone_number;

    cout << "\n\n Reference number: " << person_details[1].ref_number << endl;

    cout << "\n\tYour booked room number(s) is/are : \n\n";

    for (int k = 0; k < 10; k++){
            if (person_book[1].booked_room[k] != 0){
            cout << person_book[1].booked_room[k] << " ";
          }
    }

    cout << "\n\n Check in date: " << person_details[1].day1 << "." << person_details[1].month1 << "." << person_details[1].year1 << endl;
    cout << "\n Check out date: " << person_details[1].day2 << "." << person_details[1].month2 << "." << person_details[1].year2 << endl;

    double days = date(person_details[1].day1, person_details[1].month1, person_details[1].year1, person_details[1].day2, person_details[1].month2,person_details[1].year2);

    if (person_book[1].room_size == 1 && person_book[1].number_of_rooms == 2){
            cout << "\nTotal price: " << (100*days*2*90)/100 << " euro (10% included)\n\n";
        }else if (person_book[1].room_size == 2 && person_book[1].number_of_rooms == 2){
            cout << "\nTotal price: " << (150*days*2*95)/100 << " euro (10% included)\n\n";
        } else if (person_book[1].room_size == 1) {
            cout << "\nTotal price: " << person_book[1].number_of_rooms*100*days << " euro\n\n";
        } else if (person_book[1].room_size == 2) {
            cout << "\nTotal price: " << person_book[1].number_of_rooms*150*days << " euro\n\n";
        } 

    for (int i = 2; i <= num; i++){
        
        if (person_details[i-1].user_input_two == "No" || person_details[i-1].user_input_two == "no"){
            cout << "\n\t Name: ";
            cout << person_details[i].name;
            cout << "\n\t Surname: ";
            cout <<person_details[i].surname;
            cout << "\n\t Passport number: ";
            cout << person_details[i].passport_number;
            cout << "\n\t Age: ";
            cout << person_details[i].age;
            cout << "\n\t Phone number: ";
            cout << person_details[i].phone_number;

            cout << "\n\n Reference number: " << person_details[i].ref_number << endl;

            
        }

        cout << "\n\n\tYour booked room number(s) is/are : \n\n";

        for (int k = 0; k < 10; k++){
            if (person_book[i].booked_room[k] != 0){
            cout << person_book[i].booked_room[k] << " ";
          }
        }

        cout << "\n\n Check in date: " << person_details[i].day1 << "." << person_details[i].month1 << "." << person_details[i].year1 << endl;
        cout << "\n Check out date: " << person_details[i].day2 << "." << person_details[i].month2 << "." << person_details[i].year2 << endl;

        double days = date(person_details[i].day1, person_details[i].month1, person_details[i].year1, person_details[i].day2, person_details[i].month2,person_details[i].year2);

        if (person_book[i].room_size == 1 && person_book[i].number_of_rooms == 2){
            cout << "\nTotal price: " << (100*days*2*90)/100 << " euro (10% included)\n\n";
        }else if (person_book[i].room_size == 2 && person_book[i].number_of_rooms == 2){
            cout << "\nTotal price: " << (150*days*2*95)/100 << " euro (10% included)\n\n";
        } else if (person_book[i].room_size == 1) {
            cout << "\nTotal price: " << person_book[i].number_of_rooms*100*days << " euro\n\n";
        } else if (person_book[i].room_size == 2) {
            cout << "\nTotal price: " << person_book[i].number_of_rooms*150*days << " euro\n\n";
        } 

        
        cout << "\n\n";

    }

    cout << "Continue booking? Yes/No" << endl;
    string answ;
    cin >> answ;

    if (answ == "Yes" || answ == "yes"){
        goto start;
    }

}