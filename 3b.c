#include<stdio.h>
#include<math.h>

float function_(float x){
    return(2000*log(140.0/(140-2.1*x))-9.8*x);
}

void main(){
    int a = 8;
    int b = 30;
    int step_size = 0.05;

    float temp = 0 ;
    for(int i = 1 ; i < 440 ;i++ ){
        if(a+0.05*(i) > b){
            break;
        }
        if(i%2 == 1){
            temp = temp+2*(function_(a+0.05*(i)));
        }
        else{
            temp = temp+4*(function_(a+0.05*(i)));
        }
    }

    float fx0 = function_(a);
    float fx440 = function_(b);
    temp = temp + fx0 + fx440  ;

    printf("The answer is %f", (temp * (0.05))/3);
}