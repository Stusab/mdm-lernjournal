ÿþ#   W o c h e   2      C o n t a i n e r   &   D o c k e r   B a s i c s 
 
 
 
 # #   I n h a l t e   d e r   W o c h e 
 
 
 
 I n   W o c h e   2   h a b e   i c h   m i c h   m i t   d e n   G r u n d l a g e n   v o n   D o c k e r   b e s c h ä f t i g t .   I c h   h a b e   g e l e r n t ,   w i e   m a n   e i n   I m a g e   a u s   d e m   D o c k e r   H u b   h e r u n t e r l ä d t ,   d a r a u s   e i n e n   C o n t a i n e r   e r s t e l l t ,   d i e s e n   s t a r t e t   u n d   ü b e r   d e n   B r o w s e r   d a r a u f   z u g r e i f t .   
 
 
 
 A l s   p r a k t i s c h e s   B e i s p i e l   h a b e   i c h   d a s   o f f i z i e l l e   I m a g e   v o n   * * M e d i a W i k i * *   v e r w e n d e t   u n d   e r f o l g r e i c h   e i n e   W e b - A n w e n d u n g   i n   e i n e m   C o n t a i n e r   l o k a l   a u s g e f ü h r t . 
 
 
 
 # #   V o r g e h e n 
 
 
 
 1 .   I c h   h a b e   ü b e r   d a s   T e r m i n a l   d a s   a k t u e l l e   M e d i a W i k i - I m a g e   m i t   f o l g e n d e m   B e f e h l   g e l a d e n : 
 
 
 
         ` ` ` b a s h 
 
         d o c k e r   p u l l   m e d i a w i k i : 1 . 4 2 . 3 
 
         ` ` ` 
 
 
 
 2 .   A n s c h l i e ß e n d   h a b e   i c h   d e n   C o n t a i n e r   m i t   d i e s e m   B e f e h l   g e s t a r t e t : 
 
 
 
         ` ` ` b a s h 
 
         d o c k e r   r u n   - - n a m e   w i k i - c o n t a i n e r   - p   8 0 8 1 : 8 0   - v   m e d i a w i k i : / v a r / w w w / h t m l   - d   m e d i a w i k i : 1 . 4 2 . 3 
 
         ` ` ` 
 
 
 
         D a d u r c h   w u r d e   M e d i a W i k i   a u f   ` h t t p : / / l o c a l h o s t : 8 0 8 1 `   v e r f ü g b a r   g e m a c h t . 
 
 
 
 3 .   I m   B r o w s e r   e r s c h i e n   d i e   M e d i a W i k i - W e b o b e r f l ä c h e   m i t   d e m   H i n w e i s ,   d a s s   d i e   ` L o c a l S e t t i n g s . p h p `   n o c h   n i c h t   e i n g e r i c h t e t   i s t      e i n   t y p i s c h e s   Z e i c h e n   d a f ü r ,   d a s s   d i e   A n w e n d u n g   k o r r e k t   l ä u f t   u n d   b e r e i t   z u r   K o n f i g u r a t i o n   i s t . 
 
 
 
 4 .   M i t   d e m   B e f e h l   ` d o c k e r   p s `   k o n n t e   i c h   m i r   a n z e i g e n   l a s s e n ,   d a s s   d e r   C o n t a i n e r   e r f o l g r e i c h   l ä u f t . 
 
 
 
 # #   S c r e e n s h o t s 
 
 
 
 # # #   M e d i a W i k i - W e b o b e r f l ä c h e   i m   B r o w s e r 
 
 
 
 ! [ M e d i a W i k i   i m   B r o w s e r ] ( . . / i m a g e s / w o c h e 0 2 _ m e d i a w i k i - b r o w s e r . p n g ) 
 
 
 
 # # #   T e r m i n a l a u s g a b e   m i t   ` d o c k e r   p s ` 
 
 
 
 ! [ D o c k e r   P S ] ( . . / i m a g e s / w o c h e 0 2 _ d o c k e r - p s . p n g ) 
 
 
 
 # #   E r k e n n t n i s s e 
 
 
 
 I c h   h a b e   d u r c h   d i e s e   Ü b u n g   v e r s t a n d e n ,   w i e   s c h n e l l   s i c h   f e r t i g e   A n w e n d u n g e n   m i t   D o c k e r   l o k a l   s t a r t e n   l a s s e n .   B e s o n d e r s   h i l f r e i c h   w a r   z u   s e h e n ,   w i e   d i e   K o m b i n a t i o n   v o n   I m a g e ,   C o n t a i n e r ,   V o l u m e   u n d   P o r t w e i t e r l e i t u n g   z u s a m m e n s p i e l t ,   u m   e i n e   A n w e n d u n g   d i r e k t   i m   B r o w s e r   v e r f ü g b a r   z u   m a c h e n . 
 
 
 
 D e r   A n s a t z    B u i l d   o n c e ,   r u n   a n y w h e r e    i s t   n u n   f ü r   m i c h   g r e i f b a r .   I c h   s e h e   d a r i n   g r o ß e s   P o t e n z i a l ,   u m   E n t w i c k l u n g s -   u n d   D e p l o y m e n t p r o z e s s e   z u   v e r e i n f a c h e n . 
 
 
 
 # #   A u s b l i c k 
 
 
 
 I n   d e r   n ä c h s t e n   W o c h e   m ö c h t e   i c h   m i c h   m i t   d e m   S p e i c h e r n   u n d   B e r e i t s t e l l e n   v o n   M a c h i n e   L e a r n i n g   M o d e l l e n   b e s c h ä f t i g e n .   Z u s ä t z l i c h   p l a n e   i c h ,   m i c h   m i t   * * D o c k e r   C o m p o s e * *   a u s e i n a n d e r z u s e t z e n ,   u m   m e h r e r e   C o n t a i n e r   ( z . / B .   A p p   u n d   D a t e n b a n k )   g e m e i n s a m   z u   v e r w a l t e n . 
 
 

## Docker Compose Deployment

Um die Containerverwaltung zu vereinfachen, habe ich zusätzlich ein `docker-compose.yml` erstellt. Es enthält zwei Services: MediaWiki (Frontend) und MariaDB (Datenbank).

```yaml
version: '3.3'

services:
  db:
    image: mariadb:10.5
    restart: always
    environment:
      MYSQL_DATABASE: mediawiki
      MYSQL_USER: wikiuser
      MYSQL_PASSWORD: wikisecret
      MYSQL_ROOT_PASSWORD: rootpass
    volumes:
      - db_data:/var/lib/mysql

  mediawiki:
    image: mediawiki:1.39
    restart: always
    ports:
      - "8080:80"
    environment:
      MEDIAWIKI_DB_TYPE: mysql
      MEDIAWIKI_DB_HOST: db
      MEDIAWIKI_DB_NAME: mediawiki
      MEDIAWIKI_DB_USER: wikiuser
      MEDIAWIKI_DB_PASSWORD: wikisecret
    depends_on:
      - db

volumes:
  db_data:
```

Die Container wurden mit folgendem Befehl gestartet:

```bash
docker-compose up -d
```

Anschließend war MediaWiki unter `http://localhost:8080` erreichbar.

### Docker Compose Terminalausgabe

![docker-compose-up](../images/b3ebf40b-3152-4bcb-8794-878d5047e257.png)

Durch `docker-compose` wurde mir bewusst, wie effektiv man mehrere abhängige Dienste orchestrieren kann. Es erleichtert die Verwaltung deutlich gegenüber der manuellen Einzelkonfiguration.