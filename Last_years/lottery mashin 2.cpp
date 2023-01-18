#include <iostream>
using namespace std;

/* Make a command line application that draws lottery numbers. 7 numbers and 3 bonus numbers.
The numbers are between 1 and 39. At the end, the application will sort the numbers from biggest to smallest.
First the regular numbers and then the bonus numbers.
You can also include lottery tickets from few users and define prize money for different results.
For example, 5 numbers correct, 5+1 correct, 6 correct, 6+1 correct and all seven correct.
You can also add a feature that draws random lottery numbers for the user.
After the actual lottery draw the application tells how many numbers were correct and how much money the user won.
Text based user interface is defined by the student.*/

struct Numbers_struct {
    int num1;
    int num2;
    int user_num;
};


int main(){

    Numbers_struct numbers[] = {};
    int max = 0;

    cout << "Hello, \n This is a lottery game where you can win money ;) or loose everything :( " << endl;
    
    for (int i = 0; i < 7; i++){
        numbers[i].num1 = (rand()%39) + 1;  // creating random lottery numbers
    }

    cout << endl;
    
    for (int i = 0; i < 3; i++){
        numbers[i].num2 = (rand()%39) + 1; // creating random bonus numbers
    }


    
    int random_lottery_numbers[7] = {0,0,0,0,0,0,0};

    for (int i = 0; i < 7; i++){
         random_lottery_numbers[i] = numbers[i].num1; // put randomazed lottery numbers to another array, otherwise random numbers will be changed
    }
    

    int random_bonus_numbers[3] = {0,0,0};

    for (int i = 0; i < 3; i++){
         random_bonus_numbers[i] = numbers[i].num2; // put randomazed bonus numbers to another array, otherwise random numbers will be changed
    }
    
    cout << endl;
    
    string user_answ;  
    cout << "Do you want to play ?";  // asking a user if he\she wants to play
    cin >> user_answ;
    if (user_answ == "Yes" && user_answ == "yes"){
        cout << "Good!";
    } else {
        
    }

    cout << endl;
    cout << endl;
    cout << "Seven Lottery numbers and plus three Bonus numbers have been generated. \nIf you guess 7 numbers, you will win 1.000.000$ \nIn case of 6, you will win 100.000$ \nIn case of 5, you will win 10.000$ \nIn case of 4, you will win 1.000$ \nIn case of 3, you will win 100$ \nIn case of 2, you will win 5$ \nIn case of 1, you will win 1$ \nIn case of 0, you will win 0$" << endl;
    cout << "Your numbers must match with 7 lottery numbers, but if you do not have a match with some numbers, then the program will check them with 3 Bonus numbers" << endl;
    cout << "ENTER 7 numbers: " << endl;
    Numbers_struct user_numbers[7];

    for (int i = 0; i < 7; i++){
        cin >> user_numbers[i].user_num; // asking for user input / 7 numbers
    }

    int a = 0;
    int user_win_one[7] = {0,0,0,0,0,0,0}; 
    for (int i = 0; i < 7; i++){                  // compare user's lottery numbers with randoms and define simalar ones
        for (int k = 0; k < 7; k++){
            if (user_numbers[i].user_num == random_lottery_numbers[k]){ 
                user_win_one[i] = random_lottery_numbers[k];
                a = a + 1;
                break;
            } else{
                continue;
            }
            
        }
    }
    cout << "You correct numbers 1: " << endl;
    for (int i = 0; i < 7; i++){
        if (user_win_one[i] != 0 ){
            cout << user_win_one[i] << endl;
        }
    }


    cout << endl;

    int b = 0;
    int user_win_two[7] = {0,0,0,0,0,0,0};
    for (int i = 0; i < 7; i++){                  // compare user's bonus numbers with randoms and define simalar ones
        for (int k = 0; k < 3; k++){
            if (user_numbers[i].user_num == random_bonus_numbers[k]){
                user_win_two[i] = random_bonus_numbers[k];
                b = b + 1;
                break;
            } else{
                continue;
            }
            
        }
    }
    cout << "You correct numbers 2: " << endl;
    for (int i = 0; i < 7; i++){
        if (user_win_two[i] != 0 ){
            cout << user_win_two[i] << endl;
        }
    }


    int sum_of_wins_numbers = a + b;

    if (sum_of_wins_numbers == 7){
        cout << "Congratulations!, you've won 1.000.000$ !!!!" << endl;
    } else if (sum_of_wins_numbers == 6){
        cout << "Congratulations!, you've won 100.000$ !!!!" << endl;
    } else if (sum_of_wins_numbers == 5){
        cout << "Congratulations!, you've won 10.000$ !!!!" << endl;
    } else if (sum_of_wins_numbers == 4){
        cout << "Congratulations!, you've won 1.000$ !!!!" << endl;
    } else if (sum_of_wins_numbers == 3){
        cout << "Congratulations!, you've won 100$ !!!!" << endl;
    } else if (sum_of_wins_numbers == 2){
        cout << "Congratulations!, you've won 5$ !!!!" << endl;
    } else if (sum_of_wins_numbers == 1){
        cout << "Congratulations!, you've won 1$ !!!!" << endl;
    } else if (sum_of_wins_numbers == 0){
        cout << "Unfortunately, you have not won anything" << endl;
    }

    cout << endl;

    cout << "Printing nambers in order" << endl;

    for (int i = 0; i < 3; i++ ){
        for (int k = 0; k < 3 - i - 1; k++){
            if (random_bonus_numbers[k] > random_bonus_numbers[k + 1]){
                max = random_bonus_numbers[k];
                random_bonus_numbers[k] = random_bonus_numbers[k + 1];
                random_bonus_numbers[k + 1]= max;}
        }
    }
    
    for (int i = 0; i < 7; i++){
        for (int j = 0; j < 7 - i - 1; j++){
            if (random_lottery_numbers[j] > random_lottery_numbers[j + 1]) {
            max = random_lottery_numbers[j];
            random_lottery_numbers[j] = random_lottery_numbers[j + 1];
            random_lottery_numbers[j + 1] = max;
           }
        }
    }

    for (int i = 0; i < 7; i++){
        cout << random_lottery_numbers[i] << endl;
    }
    cout << endl;
    for (int i = 0; i < 3; i++){
        cout << random_bonus_numbers[i]  << endl;
    }


    return 0;
}