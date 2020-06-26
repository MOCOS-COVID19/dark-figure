#
# Load libraries
library(rms)
library(DALEX)
library(dplyr)

#
# Read data
data_age <- read.table("addresses_ages.csv", sep = ",", header = TRUE)
susceptibles <- read.table("susceptibles_by_age_full.csv", sep = ",", header = TRUE)[1,-1]

#
# set seed
set.seed(1)

# only non DPS and  contacts_of_the_same_address <= 15
data_age <- filter(data_age, count_contacts_of_the_same_address <= 15, is_dps == "False")

# distribution of age in secondary cases
age_secondary <- get_clean_vector(data_age$later_ages_in_address)

# distribution of age in secondary severe cases
age_secondary_sev10 <- get_clean_vector(data_age$later_ages_in_address_severe10)
age_secondary_sev14 <- get_clean_vector(data_age$later_ages_in_address_severe14)

# distribution of age in secondary dead cases
age_secondary_dead <- get_clean_vector(data_age$later_ages_in_address_deaths)

# combine dead and severe
age_secondary_sev10_dead <- c(age_secondary_sev10,age_secondary_dead)
age_secondary_sev14_dead <- c(age_secondary_sev14,age_secondary_dead)

# restricted to interval 20-80
lrm_alpha_10 <- get_alpha_lrm(pmax(pmin(80, age_secondary), 20), 
                              pmax(pmin(80, age_secondary_sev10_dead), 20),
                              label = "alpha_10")
lrm_alpha_14 <- get_alpha_lrm(pmax(pmin(80, age_secondary), 20), 
                              pmax(pmin(80, age_secondary_sev14_dead), 20),
                              label = "alpha_14")

# 
# get explainers for beta
susceptibles_df <- data.frame(age = 20:80, 
                              y = c(sum(susceptibles[1:21]), 
                                    unlist(susceptibles[22:80]),
                                    sum(susceptibles[81:101])))

lrm_beta_10 <- get_beta_lrm(susceptibles_df, 
                            pmax(pmin(80, age_secondary_sev10_dead), 20), 
                            "beta_10")
lrm_beta_14 <- get_beta_lrm(susceptibles_df, 
                            pmax(pmin(80, age_secondary_sev14_dead), 20), 
                            "beta_14")

# plot the explainer
plot(model_profile(lrm_beta_10, variables = "age")$agr_profiles,
     model_profile(lrm_beta_14, variables = "age")$agr_profiles,
     model_profile(lrm_alpha_10, variables = "age")$agr_profiles,
     model_profile(lrm_alpha_14, variables = "age")$agr_profiles) +
  geom_vline(xintercept = c(20,40,60,80), color = "grey", lty=2) +
  ggtitle("Severity rate (log scale) for secondary household infections","") + ylab("") + 
  scale_y_log10(breaks = c(0.001,0.002,0.003,0.005,
                           0.01,0.02,0.03,0.05,
                           0.1,0.2,0.3,0.5)) +
  scale_x_continuous("age",breaks = c(0,20,40,60,80)) +
  theme(panel.grid.minor = element_blank(), legend.position = "none") +facet_null()












# helper functions

get_alpha_lrm <- function(age_secondary, age_secondary_sev_dead, label = "alpha") {
  # remove sev_dead from all cases
  df <- rbind(data.frame(age = age_secondary, group = "all"),
              data.frame(age = age_secondary_sev_dead, group = "sev_dead"))
  tmp <- table(df$age, df$group)
  tmp[,1] <- tmp[,1] - tmp[,2] # here we substract dead and sev
  # decompress this for modeling
  df <- rbind(data.frame(age = as.numeric(rep(rownames(tmp), times = tmp[,1])), group = "none"),
              data.frame(age = as.numeric(rep(rownames(tmp), times = tmp[,2])), group = "sev_dead"))
  df$group <- factor(df$group)
  
  # create model for alpha
  model <- lrm(group == "sev_dead" ~ rcs(age), data = df)
  
  # create an explainer for alpha
  lrm_exp <- explain(model, data = df, y = df$group == "sev_dead", label = label)
  lrm_exp
} 

get_beta_lrm <- function(susceptibles_df, age_secondary_sev_dead, label = "beta") {
  tab <- table(age_secondary_sev_dead)
  df <- rbind(data.frame(age = rep(susceptibles_df$age, susceptibles_df$y), 
                    group = "all"),
              data.frame(age = rep(as.numeric(names(tab)), 
                                   c(tab)),
                         group = "sev"))
  
  df$group <- factor(df$group)
  modelBeta <- lrm(group == "sev" ~ rcs(age), data = df)
  lrm_exp_beta <- explain(modelBeta, 
                          data = df, 
                          y = df$group == "sev",
                          label = label)
}

get_clean_vector <- function(later_ages) {
  age_secondary_sev <- gsub(unlist(strsplit(as.character(later_ages), split = ",")), 
                            pattern = "[^0-9\\.]", replacement = "")
  na.omit(as.numeric(age_secondary_sev))
}

