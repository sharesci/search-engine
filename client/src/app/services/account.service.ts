import { Injectable } from '@angular/core'
import { Http, Response, Headers, RequestOptions, URLSearchParams } from '@angular/http'
import { Observable } from 'rxjs';
import { AppConfig } from '../app.config.js';
import { User } from '../entities/user.entity.js';
import 'rxjs/add/operator/map';

@Injectable()
export class AccountService {

    constructor(private _http: Http, private _config: AppConfig) { }

    private _createAccountUrl = this._config.apiUrl + "/account/create";
    private _getUserInfoUrl = this._config.apiUrl + "/api/v1/userinfo?";

    create(user: User): Observable<any> {
        let data = new URLSearchParams();
        
        data.append('username', user.username);
        data.append('password', user.password);
        data.append('firstname', user.firstname);
        data.append('lastname', user.lastname);
        data.append('institution', user.institution);
        data.append('self_bio', user.self_bio);

        return this._http.post(this._createAccountUrl, data)
            .map((response: Response) => response.json());
    }

    getUserInfo(username: string) {
        let queryString = new URLSearchParams();
        queryString.append('username', username);

        return this._http.get(this._getUserInfoUrl + queryString.toString())
            .map((response: Response) => response.json());
    }
}