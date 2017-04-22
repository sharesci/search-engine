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
    private isUserLoggedInSource = new Subject<boolean>();
    isUserLoggedIn$ = this.isUserLoggedInSource.asObservable();

    login(username: string, password: string): Observable<any> {
        let data = new URLSearchParams();
        data.append('username', username);
        data.append('password', password);

        return this._http.post(this._loginUrl, data)
            .map((response: Response) => {
                this.isUserLoggedInSource.next(true); 
                return response.json();
            });
    }

    logout() {
        let observ = this._http.post(this._logoutUrl, {});
        observ.subscribe(null, null, () => { this.isUserLoggedInSource.next(false)});
        return observ;
    }
}