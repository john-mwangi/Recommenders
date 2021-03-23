## ----setup, include=FALSE-----
knitr::opts_chunk$set(echo = TRUE)


## -----------------------------
library(recommenderlab)
library(tidyverse)


## -----------------------------
data("MSWeb")
data("MovieLense")
data("Jester5k")


## -----------------------------
class(MSWeb)
class(MovieLense)
class(Jester5k)

# DATA CHECKS

## -----------------------------
dim(MSWeb)

## -----------------------------
rownames(MSWeb)[1:10] #user ids
colnames(MSWeb)[1:10] #product names


## -----------------------------
ms_purchases <- as(MSWeb, "list") %>% 
  tibble(user_id = names(.)) %>% 
  unnest(cols = ".") %>% 
  rename(purchase = ".") %>% 
  count(user_id,purchase)


## -----------------------------
ms_purchases %>% 
  filter(n>1)


## -----------------------------
as(MovieLense,"list") %>% 
  tibble() %>% 
  rename(ratings = ".") %>% 
  mutate(len = map_int(.x = ratings, .f = length),
         idx = row_number()) %>% 
  arrange(desc(len))


## -----------------------------
as(MovieLense,"list")[[405]] %>% 
  data.frame() %>% 
  rownames_to_column() %>% 
  count(rowname) %>% 
  arrange(desc(n))

# SIMPLE STATS

## Summary info

## -----------------------------
summary(rowCounts(MSWeb))

## Histogram

## -----------------------------
which(grepl(x = rownames(MSWeb), pattern = "^1$"))

as(MSWeb[1:2,], "list")

hist(rowCounts(MSWeb))

## Minimum cut-off

## -----------------------------
table(rowCounts(MSWeb)) %>% 
  data.frame()


## -----------------------------
table(rowCounts(MSWeb)) %>% 
  data.frame() %>% 
  mutate(total_purchases = sum(Freq)) %>% 
  mutate(Var1 = as.integer(Var1)) %>% 
  filter(Var1>=10) %>% 
  mutate(purchases_10 = sum(Freq)) %>% 
  summarise(purchases_10 = mean(purchases_10),
            total_purchases = mean(total_purchases)) %>% 
  mutate(prop = purchases_10/total_purchases)


## -----------------------------
MSWeb10 <- MSWeb[rowCounts(MSWeb)>=10,]

dim(MSWeb10)


## -----------------------------
summary(rowCounts(MSWeb10))

# CROSS-VALIDATED TRAINING

## Evaluation scheme

## -----------------------------
es <- 
evaluationScheme(data = MSWeb10, 
                 method = "cross-validation",
                 train = 0.8,
                 k = 5,
                 given = -1)

## List of recommenders
## -----------------------------
algorithms <- list(
  popular = list(name = "POPULAR", param = NULL),
  ubcf = list(name = "UBCF", param = NULL),
  ibcf = list(name = "IBCF", param = NULL),
  ar = list(name = "AR", param = NULL),
  hybrid = list(name = "HYBRID", param =
      list(recommenders = list(
          popular = list(name = "POPULAR", param = NULL),
          ubcf = list(name = "UBCF", param = NULL),
          ibcf = list(name = "IBCF", param = NULL)
        )
      )
  )
)

## Cross-validated Training
## -----------------------------
system.time({
  
ev_list <- 
evaluate(x = es,
         method = algorithms,
         type = "topNList",
         n = c(1,3,5,10))

})

## RO Curves
## -----------------------------
names(ev_list)


## -----------------------------
plot(ev_list, legend="topleft", annotate=TRUE, main="ROC Curves of each model")

plot(ev_list[[1]], annotate=TRUE, main="ROC Curve of Popularity Model")

## PR Curves
## -----------------------------
plot(x = ev_list, y = "prec/rec", annotate=TRUE, legend="topleft")

## Area Under Curve
## -----------------------------
names(ev_list)

ibcf_eval <- avg(ev_list$ibcf) %>% 
  data.frame() %>% 
  arrange(recall)

pop_eval <- avg(ev_list$popular) %>% 
  data.frame() %>% 
  arrange(recall)

pracma::trapz(x = ibcf_eval$recall, y = ibcf_eval$recall)
pracma::trapz(x = pop_eval$recall, y = pop_eval$recall)

# TRAIN RECOMMENDERS
## -----------------------------
ke_ubcf = Recommender(data = MSWeb10, method = "UBCF")
ke_pop = Recommender(data = MSWeb10, method = "POPULAR")
ke_hyb = HybridRecommender(ke_ubcf, ke_pop)

# PREDICTIONS

## Raw predictions
## -----------------------------
pre = predict(ke_pop, MSWeb10[100:101], n=10)
# We can also use 5 because we saw it is the best threshold

rownames(MSWeb10[100:101])

pre = as(pre, "list")

pre

## Select specific user ids
## -----------------------------
req_user_id <- c(3097,3164)

#rec_user_id <- c("^3097$","^3164$")

records <- c()
for(id in seq_along(req_user_id)){
  records <- append(records,paste0("^",req_user_id,"$"))
}

records <- unique(records)

rec_nums <- which(grepl(x = rownames(MSWeb10),
            pattern = paste0(records, collapse = "|")))

pre_flex <- predict(object = ke_pop, 
                    newdata = MSWeb10[rec_nums], 
                    n=5)

as(pre_flex, "list")

## Format into a data frame
## -----------------------------
as(pre_flex,"list") %>% 
  data.frame() %>% 
  unnest() %>% 
  pivot_longer(cols = everything()) %>% 
  mutate(name = str_remove(name,"^X")) %>% 
  arrange(name)

## Extracting ratings
## -----------------------------
pred_ratings <- predict(ke_pop, MSWeb10[rec_nums], type="ratings")

pred_ratings_list <- as(pred_ratings,"list")

item_name_vec <- c()
for(user in seq_along(pred_ratings_list)){
  item_names <- names(pred_ratings_list[[user]])
  item_name_vec <- append(item_name_vec, item_names)
}

preds_all <- pred_ratings_list %>% 
  tibble(user_id = names(.)) %>% 
  unnest(cols = ".") %>% 
  mutate(item = item_name_vec) %>% 
  rename(score = ".") %>% 
  mutate(business = sample(x = c("life","non-life","health","asset","banking"),
                           size = 541,
                           replace = TRUE,
                           prob = c(0.1,0.2,0.4,0.15,0.15))) %>% 
  select(user_id,item,business,score)

preds_all


## -----------------------------
preds_by_business <- preds_all %>% 
  group_by(user_id, business) %>% 
  arrange(desc(score), .by_group = TRUE) %>% 
  slice(1:3) %>% 
  ungroup()

preds_by_business


## -----------------------------
top_5 <- preds_all %>% 
  group_by(user_id) %>% 
  arrange(desc(score), .by_group = TRUE) %>% 
  slice(1:5)

top_5


## -----------------------------
 writexl::write_xlsx(list(by_business = preds_by_business, 
                          top_5 = top_5), 
                     "./pred_list_score.xlsx")

# HYBRID RECOMMENDER OPTIMISATION (RANDOM SEARCH)

## Sample 5 different combinations of weights
## -----------------------------
rec_weights <- crossing(m1 = seq(0,1,0.1),
         m2 = seq(0,1,0.1)) %>% 
  mutate(weights = paste0(m1,",",m2)) %>% 
  pull(weights)

set.seed(123)
rec_weights_sample <- sample(rec_weights, 5)


## -----------------------------
recommenderRegistry$get_entry_names()

## Test with some weights
## -----------------------------
hybrids <- 
list(hybrid = list(name = "HYBRID", 
                   param = list(recommenders = list(popular = list(name = "POPULAR", 
                                                                   param = NULL),
                                                    ubcf = list(name = "UBCF", 
                                                                param = NULL)),
                                weights = c(0.2,0.8))))

## List of hybrids with different weights
## -----------------------------
hybrids_list <- list()
for (w in seq_along(rec_weights_sample)){
hybrids_list[[w]] <- list(hybrid = list(name = "HYBRID", 
                   param = list(recommenders = list(popular = list(name = "POPULAR", 
                                                                   param = NULL),
                                                    ubcf = list(name = "UBCF", 
                                                                param = NULL)),
                                weights = as.numeric(strsplit(rec_weights_sample[[w]],",")[[1]])
                                )))
  
}

## Training
## -----------------------------
ev_hybrid <- list()
for(h in seq_along(hybrids_list)){
  ev_hybrid[[h]] <- evaluate(es, hybrids_list[[h]], n = c(1,3,5,10))
}

## Evaluation results
## -----------------------------
class(ev_hybrid)

class(ev_hybrid[[1]])

## -----------------------------
for(p in seq_along(ev_hybrid)){
  plot(x = ev_hybrid[[p]], y = "prec/rec")
}

## Sense check
## -----------------------------
# Check hybrid 01
test_ev <- evaluate(es, hybrids, n = c(1,3,5,10))

plot(test_ev, y = "prec/rec")

# COLD START
## Popular products

## -----------------------------
ms_purchases %>% 
  count(purchase) %>% 
  left_join(preds_all, by=c("purchase"="item")) %>% 
  select(purchase,business,n) %>% 
  arrange(desc(n)) %>% 
  filter(!is.na(business)) %>% 
  group_by(business) %>% 
  slice(1:3)


## -----------------------------
save.image("./Showcase - dummy.RData")

