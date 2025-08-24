const express = require('express');
const axios = require('axios');
const app = express()

const PORT = 3000

app.get('/feed', async (req, res) => {
    try {
        const flaskApiUrl = "127.0.0.1:5000/live-feed";
        const response = await axios.get(flaskApiUrl);
        res.header('Content-Type', 'multipart/x-mixed-replace; boundary=frame');
        res.write('--frame\r\nContent-Type: image/jpeg\r\n\r\n');
        res.write(response.data);
        res.write('\r\n\r\n');
        res.on('close', () => {
            console.log("Feed closed")
        });


    } catch (e) {
        console.log("Error fetching feed:", e.message)
        res.status(500).send("Internal server error")
    }

});


app.listen(PORT, "0.0.0.0", () => {
    console.log("Server is up and running listening on port:", PORT)

})