#include <stdio.h>


int main(void) {
  double val;
  int i;
  for (i=0; i<10; i++) {
    val = (double)rand()/(double)1;
    printf("%f ", val);
  }
  return 0;
}
