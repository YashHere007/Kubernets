frontend -> backend -> db 

creating docker images for frontend, backend.

orchestrating these images with kubernetes.

Steps to run the project

kubectl delete -f k8s/
kubectl get all

docker build -t localhost/backend-image ./backend
docker build -t localhost/frontend-image ./frontend

kubectl apply -f k8s/

kubectl get all

kubectl exec -it <postgres-podname> -- psql -U postgres -d mydb -c "CREATE TABLE users (id SERIAL PRIMARY KEY, username VARCHAR(50) UNIQUE NOT NULL, password VARCHAR(50) NOT NULL);INSERT INTO users (username, password) VALUES ('user1', 'password123');"

Terminal#1
kubectl port-forward svc/frontend-service 8080:80 

Terminal#2
kubectl port-forward svc/backend-service 5000:5000
