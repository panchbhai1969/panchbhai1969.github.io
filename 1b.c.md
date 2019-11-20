#include<stdio.h>



float lagrangian_weights(int number_of_points, int x, int xi, int count ){
    float data[3] = {0,10.0,15.0};
    float temp = 1;
    for(int i = 0; i < number_of_points ; i++ )
    {
        if(count == i){
            continue;
        }
        temp = temp * ((x-data[i])/(xi-data[i]));
        //printf("%f\n",temp);
    }
    printf("Lagrangian : %f\n", temp);
    return temp;
}

/* void function_(x){
    return velocity[x*10];
} */

void main(){
    float velocity[301] = {0};
    velocity[0] = 0;
    velocity[100] = 227.04;
    velocity[150] = 362.78;
    velocity[200] = 517.35;
    velocity[225] = 602.97;
    velocity[300] = 901.67;
    
    float data[3] = {0,10,15};
    float x_o_i = 14.0;
    float temp = 0;
    for(int i = 0; i < 3;i++){
        
        int index = data[i]*10;
        temp = temp + lagrangian_weights(3, x_o_i, data[i], i) * velocity[index];
        printf("Printing temp: %f\n", temp);
    }

    printf("%f\n",temp);

}