import { Injectable } from '@angular/core'
import { Http, Response, Headers, RequestOptions, URLSearchParams } from '@angular/http'
import { Observable } from 'rxjs';
import { AppConfig } from '../app.config.js';
import { Subject } from 'rxjs/Subject';
import { Router } from '@angular/router';
import { SharedService } from './shared.service.js';
import 'rxjs/add/operator/map';

@Injectable()
export class AuthenticationService {

    constructor(private _http: Http, private _config: AppConfig, private _sharedService: SharedService,
        private _router: Router) { }

    private _loginUrl = this._config.apiUrl + '/login';
    private _logoutUrl = this._config.apiUrl + '/logout';

    login(username: string, password: string): Observable<any> {

        let data = new URLSearchParams();
        data.append('username', username);
        data.append('password', password);

        let observ = this._http.post(this._loginUrl, data)
            .map((response: Response) => { return response.json() })

        observ.subscribe(
            data => {
                if (data.errno == "0") {
                    localStorage.setItem('currentUser', username);
                    this._sharedService.setLoginStatus(true);
                    this._router.navigate(['/']);
                }
            },
            error => console.log(error)
        )
        return observ;
    }

    logout(): Observable<Response> {
        let observ = this._http.post(this._logoutUrl, {});
        return observ;
    }
}