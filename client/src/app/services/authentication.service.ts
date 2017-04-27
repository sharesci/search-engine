import { Injectable } from '@angular/core'
import { Http, Response, Headers, RequestOptions, URLSearchParams } from '@angular/http'
import { Observable } from 'rxjs';
import { AppConfig } from '../app.config.js';
import { Subject }    from 'rxjs/Subject';
import 'rxjs/add/operator/map';

@Injectable()
export class AuthenticationService {
    
    constructor(private _http: Http, private _config: AppConfig) { }

    private _loginUrl = this._config.apiUrl + '/login';
    private _logoutUrl = this._config.apiUrl + '/logout';

    login(username: string, password: string): Observable<any> {
        let data = new URLSearchParams();
        data.append('username', username);
        data.append('password', password);

        return this._http.post(this._loginUrl, data)
            .map((response: Response) => {                
                return response.json();
            });
    }

    logout(): Observable<Response> {
        let observ = this._http.post(this._logoutUrl, {});
        return observ;
    }
}