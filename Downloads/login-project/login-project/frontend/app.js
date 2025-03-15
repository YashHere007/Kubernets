document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const response = await fetch('http://localhost:5000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();
        document.getElementById('message').textContent = data.message;
        document.getElementById('message').style.color = response.ok ? 'green' : 'red';
    } catch (error) {
        document.getElementById('message').textContent = 'Error connecting to server';
        document.getElementById('message').style.color = 'red';
    }
});