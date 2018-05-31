review <- read.csv('Seattle/individual_reviews.csv')

# Handle hyginen code
review$violation <- rep(NA, nrow(review))
review[review$inspection_penalty_score == 0, ][, "violation"] <- 'no_violation'
review[review$inspection_penalty_score > 50, ][, "violation"] <- 'severe_violation'
review$violation[is.na(review$violation)] <- 'minor_violation'

rev_tm <- NULL
rev_tm$id <- review$review_id
rev_tm$score <- paste(review$rating, review$violation, sep = " ")
rev_tm$text <- gsub("[\r\n]", " ", review$content)
rev_tm <- as.data.frame(rev_tm)

write.table(rev_tm,"rev_tm_violation_rating.txt",row.names=FALSE, sep = "\t")
