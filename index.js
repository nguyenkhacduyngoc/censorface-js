import express from 'express';
import path from 'path';

const app = express();

// Create upload dir
const __dirname = path.resolve();
const PORT = 3000;

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "localhost"); // update to match the domain you will make the request from
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});

app.use(express.static(__dirname));

app.get('/*', function (req, res) { 
    const ip = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
    console.log(`Opening from ${ip}`); // ip address of the user
    
    res.sendFile('index.html');
});

console.log(`Listening to port ${PORT}`);

app.listen(PORT);