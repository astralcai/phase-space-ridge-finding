#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) {

  int i, j, npart;
  float info[7];
  char outname[1000];
  FILE *infile, *outfile, *outinfo;

  if (argc < 3) {
    fprintf(stderr, "Usage:\t./reverter <binary_datafile> <info_file> <outfile1>, <outfile2> ...\n");
    exit(1);
  }    

  outfile = fopen(argv[1], "r");
  outinfo = fopen(argv[2], "r");

  for (i=3; i<argc; i++) {
    infile = fopen(argv[i], "w");
    if (infile == NULL) {
      fprintf(stderr, "File not found (%s)\n", argv[i]);
      continue;
    }
    fscanf(outinfo, "%d", &npart);
    fprintf(infile, "%d %e\n", npart, 0.0);
    for (j=0; j<npart; j++) {
      fread(info, sizeof(float), 7, outfile);
      fprintf(infile, "%e %e %e %e %e %e %e\n", info[0], info[1], info[2], 
              info[3], info[4], info[5], info[6]);
    }
    fclose(infile);
  }
  fclose(outfile);
  fclose(outinfo);
}

