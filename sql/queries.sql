-- Q1: WHERE, GROUP BY and HAVING
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders WHERE status='completed' AND order_amount>500
GROUP BY customer_id HAVING SUM(order_amount)>8000
ORDER BY total_spend DESC;

-- Q2: Three-table join; completed revenue by city and category
SELECT c.city,p.category,ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o JOIN customers c ON o.customer_id=c.customer_id
JOIN products p ON o.product_id=p.product_id
WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;

-- Q3: Scalar subquery; products above average catalogue price
SELECT product_id,product_name,category,price_inr FROM products
WHERE price_inr>(SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4: LEFT JOIN; customers without completed orders
SELECT c.customer_id,c.city FROM customers c
LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5: CTEs; customers whose completed spend exceeds average customer spend
WITH customer_spend AS (
 SELECT customer_id,SUM(order_amount) AS total_spend FROM orders
 WHERE status='completed' GROUP BY customer_id
), average_spend AS (SELECT AVG(total_spend) AS avg_spend FROM customer_spend)
SELECT cs.customer_id,ROUND(cs.total_spend,2) AS total_spend
FROM customer_spend cs CROSS JOIN average_spend a
WHERE cs.total_spend>a.avg_spend ORDER BY cs.total_spend DESC;

-- Q6: RANK, DENSE_RANK and ROW_NUMBER by city
WITH customer_spend AS (
 SELECT c.city,c.customer_id,SUM(o.order_amount) AS total_spend
 FROM customers c JOIN orders o ON c.customer_id=o.customer_id
 WHERE o.status='completed' GROUP BY c.city,c.customer_id
)
SELECT city,customer_id,ROUND(total_spend,2) AS total_spend,
 RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) AS row_num
FROM customer_spend ORDER BY city,spend_rank,customer_id;

-- Q7: Daily completed revenue and cumulative running total
WITH daily_revenue AS (
 SELECT SUBSTR(order_ts,1,10) AS order_day,SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY SUBSTR(order_ts,1,10)
)
SELECT order_day,ROUND(revenue,2) AS daily_revenue,
 ROUND(SUM(revenue) OVER(ORDER BY order_day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),2) AS cumulative_revenue
FROM daily_revenue ORDER BY order_day;
