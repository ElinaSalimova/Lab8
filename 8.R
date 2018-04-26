library('tree')  
library('ISLR') 
library('randomForest') 
library('gbm')


attach(Wage)
# новая переменная
High <- ifelse(wage >= 128.68, '1', '0')
# присоединяем к таблице данных
Wage <- data.frame(Wage, High)
# модель бинарного  дерева
tree.wage <- tree(High ~ . -wage -region -logwage, Wage)
summary(tree.wage)

# график результата
plot(tree.wage)            # ветви
text(tree.wage, pretty=0)  # подписи

tree.wage                # посмотреть всё дерево в консоли

# ядро генератора случайных чисел
set.seed(6)
# обучающая выборка
train <- sample(1:nrow(Wage), 1500)
# тестовая выборка
Wage.test <- Wage[-train,]
High.test <- High[-train]

# строим дерево на обучающей выборке
tree.wage <- tree(High ~ . -wage -region -logwage, Wage, subset = train)

# делаем прогноз
tree.pred <- predict(tree.wage, Wage.test, type = "class")

# матрица неточностей
tbl <- table(tree.pred, High.test)
tbl

# оценка точности
acc.test <- sum(diag(tbl))/sum(tbl)
acc.test



# rf
wage.test <- Wage[-train, "wage"]
rf.wage <- randomForest(wage ~ . -High -region -logwage, data = Wage, subset = train,
                          mtry = 6, importance = TRUE)
# прогноз
yhat.rf <- predict(rf.wage, newdata = Wage[-train, ])
# MSE на тестовой выборке
mse.test <- mean((yhat.rf - wage.test)^2)
# важность предикторов
importance(rf.wage)  # оценки


cv.wage <- cv.tree(tree.wage, FUN = prune.misclass)
# имена элементов полученного объекта
names(cv.wage)

cv.wage

# графики изменения параметров метода по ходу обрезки дерева ###################

# 1. ошибка с кросс-валидацией в зависимости от числа узлов
par(mfrow = c(1, 2))
plot(cv.wage$size, cv.wage$dev, type = "b",
     ylab = 'Частота ошибок с кросс-вал. (dev)',
     xlab = 'Число узлов (size)')
# размер дерева с минимальной ошибкой
opt.size <- cv.wage$size[cv.wage$dev == min(cv.wage$dev)]
abline(v = opt.size, col = 'red', 'lwd' = 2)     # соотв. вертикальная прямая
mtext(opt.size, at = opt.size, side = 1, col = 'red', line = 1)

# 2. ошибка с кросс-валидацией в зависимости от штрафа на сложность
plot(cv.wage$k, cv.wage$dev, type = "b",
     ylab = 'Частота ошибок с кросс-вал. (dev)',
     xlab = 'Штраф за сложность (k)')


# дерево с 3 узлами
prune.wage <- prune.misclass(tree.wage, best = 3)

# визуализация
plot(prune.wage)
text(prune.wage, pretty = 0)

# прогноз на тестовую выборку
tree.pred <- predict(prune.wage, Wage.test, type = "class")

# матрица неточностей
tbl <- table(tree.pred, High.test)
tbl

# оценка точности
acc.test <- sum(diag(tbl))/sum(tbl)
acc.test

par(mfrow = c(1, 1))



#Бустинг----------------------
boost.wage <- gbm(wage ~ . -High -region -logwage, data = Wage[train, ], distribution = "gaussian",
                    n.trees = 5000, interaction.depth = 4)
# график и таблица относительной важности переменных
summary(boost.wage)

# графики частной зависимости для двух наиболее важных предикторов
par(mfrow = c(1, 2))
plot(boost.wage, i = "maritl")
plot(boost.wage, i = "High.1")


# прогноз
yhat.boost <- predict(boost.wage, newdata = Wage[-train, ], n.trees = 5000)

# MSE на тестовой
mse.test <- mean((yhat.boost - wage.test)^2)
mse.test


#Настройку бустинга можно делать с помощью гиперпараметра λ (аргумент shrinkage). 
#Установим его равным 0.2.

# меняем значение гиперпараметра (lambda) на 0.2 -- аргумент shrinkage
boost.wage <- gbm(wage ~ . -High -region -logwage, data = Wage[train, ], distribution = "gaussian",
                    n.trees = 5000, interaction.depth = 4, 
                    shrinkage = 0.1, verbose = F)
# прогноз
yhat.boost <- predict(boost.wage, newdata = Wage[-train, ], n.trees = 5000)

# MSE а тестовой
mse.test <- mean((yhat.boost - wage.test)^2)
mse.test
