library(tidyverse)
library(colorspace)

df = read_csv('hues.csv')

hue_to_color <- function(x){
  rgb = ((HSV(x,1,1) %>% as('RGB')) %>% attr('coords') * 255 ) %>% unlist() %>% round()
  toupper(paste0('#',paste(as.hexmode(rgb), collapse='')))
}

hue_summary = df %>%
 group_by(fn) %>%
 mutate(
   hue_prop = prop.table(cnt)
 ) %>%
 group_by(hue) %>%
 summarize(
   hue_avg = mean(hue_prop)
 ) %>%
 rowwise() %>%
 mutate(
  color_val = hue_to_color(hue)
 )

png('hue_density.png', width=1960, height=720)
print(
 ggplot(hue_summary) +
 geom_rect(
  aes(xmin=hue-0.5, xmax=hue+0.5, ymin=0, ymax=max(hue_avg)+0.01, fill=color_val)) +
 geom_bar(
  aes(x=hue, y=hue_avg),
  color='black',
  stat='identity'
 ) +
 scale_fill_identity()
)
dev.off()
