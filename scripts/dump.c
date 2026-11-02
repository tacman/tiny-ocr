/* Emit JSON for one image: OCR lines (with layout ids) and layout regions.
   cc -I$RUSTO/include dump.c -L$RUSTO/target/release -lrusto -o dump
   ./dump '<init config json>' image.png '<run options json>' */
#include <stdio.h>
#include <string.h>
#include "rusto.h"
static void esc(const char*s){putchar('"');for(;*s;s++){unsigned char c=*s;if(c=='"'||c=='\\')printf("\\%c",c);else if(c<32)printf("\\u%04x",c);else putchar(c);}putchar('"');}
int main(int argc,char**argv){
  ROCRHandle*h=rocr_initialize(argv[1]);if(!h){fprintf(stderr,"init failed\n");return 2;}
  CTextResult*r=NULL;size_t n=0;
  int rc=rocr_detect_text_file(h,argv[2],argc>3?argv[3]:"{}",&r,&n);
  printf("{\"rc\":%d,\"lines\":[",rc);
  for(size_t i=0;i<n;i++){printf("%s{\"text\":",i?",":"");esc(r[i].text);
    printf(",\"score\":%.3f,\"box\":[%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f],\"region\":%d,\"type\":%d}",
      r[i].score,r[i].box_x1,r[i].box_y1,r[i].box_x2,r[i].box_y2,r[i].box_x3,r[i].box_y3,r[i].box_x4,r[i].box_y4,r[i].region_id,r[i].layout_type);}
  printf("],\"regions\":[");
  CLayoutRegion*g=NULL;size_t m=0;
  if(rocr_detect_layout_file(h,argv[2],&g,&m)==0)
    for(size_t i=0;i<m;i++)printf("%s{\"type\":%d,\"conf\":%.2f,\"box\":[%d,%d,%d,%d]}",i?",":"",g[i].layout_type,g[i].confidence,g[i].x_min,g[i].y_min,g[i].x_max,g[i].y_max);
  printf("]}\n");return 0;}
