FROM node:alpine3.15

WORKDIR /root/

COPY server.js package.json /root/

RUN npm install .

ENTRYPOINT [ "node" ]
CMD [ "server.js" ]