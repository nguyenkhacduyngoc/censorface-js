import express from 'express';
import path from 'path';

const app = express();

// Create upload dir
const __dirname = path.resolve();
const PORT = 3000;

app.use(express.static(__dirname));

app.get('/*', function (req, res) { 
    const ip = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
    console.log(`Opening from ${ip}`); // ip address of the user
    
    res.sendFile('index.html');
});

console.log(`Listening to port ${PORT}`);

app.listen(PORT);