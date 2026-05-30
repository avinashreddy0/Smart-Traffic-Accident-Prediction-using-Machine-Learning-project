create database public
use public
SELECT * FROM smart 

-- TASK 1 
-- top 5 weather conditions with highest accident occurrences

select WEATHER_CONDITION,count(*) higest
from smart
where accident_occurrence = 1
group by weather_conditiOn
order by higest desc
limit 5

-- task2
-- average vehicle speed during accident cases

SELECT round(avg(VEHICLE_SPEED),2) as avg_vehicle_speed
from smart
where accident_occurrence = 1

-- task 3
-- total accident cases for each traffic density category

select count(*) AS TOTAL_ACCIDENT,TRAFFIC_DENSITY
FROM SMART
WHERE ACCIDENT_OCCURRENCE = 1
GROUP BY TRAFFIC_DENSITY
ORDER BY TOTAL_ACCIDENT DESC

-- TASK 4
-- top 3 road conditions with highest accident cases

SELECT COUNT(*) AS ACCIDENT_COUNT,ROAD_CONDITION 
FROM SMART
WHERE ACCIDENT_OCCURRENCE = 1
GROUP BY ROAD_CONDITION 
ORDER BY ACCIDENT_COUNT DESC
LIMIT 3

-- TASK 5 
-- average visibility during accident cases for each weather condition

SELECT round(avg(VISIBILITY),2) AS AVG_VISIBILITY,WEATHER_CONDITION
FROM SMART
WHERE ACCIDENT_OCCURRENCE = 1
GROUP BY WEATHER_CONDITION
ORDER BY AVG_VISIBILITY ASC

-- TASK 6
-- accident count for each hour of the day

SELECT COUNT(*) AS TOTAL_ACCIDENTS,HOUR
FROM SMART
WHERE ACCIDENT_OCCURRENCE = 1
GROUP BY HOUR
ORDER BY TOTAL_ACCIDENTS DESC

-- TASK 7
-- top 5 road types with highest average vehicle speed during accidents

SELECT ROUND(AVG(VEHICLE_SPEED),2) AS AVG_SPEED,ROAD_TYPE
FROM SMART
WHERE ACCIDENT_OCCURRENCE = 1
GROUP BY ROAD_TYPE
ORDER BY AVG_SPEED DESC
LIMIT 5

-- TASK 8 
-- percentage of accident cases in each weather condition

SELECT round(sum(ACCIDENT_OCCURRENCE)/COUNT(*)*100,2) AS ACCIDENT_PERCENTAGE,WEATHER_CONDITION
FROM SMART
GROUP BY WEATHER_CONDITION
ORDER BY ACCIDENT_PERCENTAGE DESC

-- TASK 9
-- total accident cases where visibility is below overall average visibility

SELECT COUNT(*) AS LOW_VISIBILITY_ACCIDENT
FROM SMART
WHERE ACCIDENT_OCCURRENCE = 1 AND VISIBILITY < (SELECT AVG(VISIBILITY) AS AVG_VISIBILITY FROM SMART)

-- TASK 10
-- weather conditions where accident percentage is greater than overall accident percentage

SELECT SUM(ACCIDENT_OCCURRENCE)/COUNT(*) * 100,WEATHER_CONDITION
FROM SMART
GROUP BY WEATHER_CONDITION 
HAVING SUM(ACCIDENT_OCCURRENCE)/COUNT(*) * 100 > (SELECT SUM(ACCIDENT_OCCURRENCE)/COUNT(*) * 100 FROM SMART)

